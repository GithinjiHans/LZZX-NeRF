import os
import time

import torch
import numpy as np

from render_3dmm import Render_3DMM
from util import *
import gc
import cv2
from progress.bar import Bar


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


batch_size = 32
device_render = torch.device("cuda:0")
device_default = torch.device("cuda:0")


class AjianUtil:
    def __init__(self):
        self.renderer:Render_3DMM = None

    def fit_焦距拟合(self, lms, sel_ids, cxy, id_dim, sel_num, exp_dim, model_3dmm):
        arg_landis = 1e5
        print('从600开始到1400，每循环一次增加100...')
        for focal in range(600, 1500, 100):
            id_para = lms.new_zeros((1, id_dim), requires_grad=True)
            exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
            euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
            trans = lms.new_zeros((sel_num, 3), requires_grad=True)
            trans.data[:, 2] -= 7
            focal_length = lms.new_zeros(1, requires_grad=False)
            focal_length.data += focal
            set_requires_grad([id_para, exp_para, euler_angle, trans])

            optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=0.1)
            optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=0.1)

            for iter in range(2000):
                id_para_batch = id_para.expand(sel_num, -1)
                # 调用模型获取几何对象
                geometry = model_3dmm.get_3dlandmarks(
                    id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
                )
                proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
                loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms[sel_ids].detach())
                loss = loss_lan
                optimizer_frame.zero_grad()
                loss.backward()
                optimizer_frame.step()
                # if iter % 100 == 0:
                #     print(focal, 'pose', iter, loss.item())
            print('   ---子循环：2000次几何对象循环完成！')

            for iter in range(2500):
                id_para_batch = id_para.expand(sel_num, -1)
                geometry = model_3dmm.get_3dlandmarks(
                    id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
                )
                proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
                loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms[sel_ids].detach())
                loss_regid = torch.mean(id_para * id_para)
                loss_regexp = torch.mean(exp_para * exp_para)
                loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
                optimizer_idexp.zero_grad()
                optimizer_frame.zero_grad()
                loss.backward()
                optimizer_idexp.step()
                optimizer_frame.step()
                # if iter % 100 == 0:
                #     print(focal, 'poseidexp', iter, loss_lan.item(), loss_regid.item(), loss_regexp.item())

                if iter % 1500 == 0 and iter >= 1500:
                    for param_group in optimizer_idexp.param_groups:
                        param_group["lr"] *= 0.2
                    for param_group in optimizer_frame.param_groups:
                        param_group["lr"] *= 0.2
            print('   ---子循环：2500次几何对象循环完成！')

            print(f'当前数:{focal},loss:{loss_lan.item()},平均值：{torch.mean(trans[:, 2]).item()}')

            if loss_lan.item() < arg_landis:
                arg_landis = loss_lan.item()
                arg_focal = focal

        print(f"[INFO] 根据最小loss，找到最好的一组数是:arg_focal={arg_focal},arg_landis={arg_landis}")
        return arg_focal, arg_landis

    def init_3DMM_render(self,arg_focal,h, w):
        self.renderer = Render_3DMM(arg_focal, h, w, batch_size, device_render)

    def fit_粗略拟合(self, arg_focal, lms, id_dim, num_frames, exp_dim, tex_dim, cxy, model_3dmm, h, w):
        print(f'[INFO] 粗略拟合...')
        # for all frames, do a coarse fitting ???
        id_para = lms.new_zeros((1, id_dim), requires_grad=True)
        exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
        tex_para = lms.new_zeros(
            (1, tex_dim), requires_grad=True
        )  # not optimized in this block ???
        euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
        trans = lms.new_zeros((num_frames, 3), requires_grad=True)
        light_para = lms.new_zeros((num_frames, 27), requires_grad=True)
        trans.data[:, 2] -= 7  # ???
        focal_length = lms.new_zeros(1, requires_grad=True)
        focal_length.data += arg_focal

        set_requires_grad([id_para, exp_para, tex_para, euler_angle, trans, light_para])

        optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=0.1)
        optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

        print('     循环1500次...')
        for iter in range(1500):
            id_para_batch = id_para.expand(num_frames, -1)
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
            )
            # print(f'    --第[{iter+1}]次生成geometry...')
            proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
            loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms.detach())
            loss = loss_lan
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_frame.step()
            if iter == 1000:
                for param_group in optimizer_frame.param_groups:
                    param_group["lr"] = 0.1
            # if iter % 100 == 0:
            #     print('pose', iter, loss.item())

        for param_group in optimizer_frame.param_groups:
            param_group["lr"] = 0.1

        print('     循环2000次...')
        for iter in range(2000):
            id_para_batch = id_para.expand(num_frames, -1)
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
            )
            # print(f'    --第[{iter + 1}]次生成geometry...')
            proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
            loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms.detach())
            loss_regid = torch.mean(id_para * id_para)
            loss_regexp = torch.mean(exp_para * exp_para)
            loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
            optimizer_idexp.zero_grad()
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_idexp.step()
            optimizer_frame.step()
            # if iter % 100 == 0:
            #     print('poseidexp', iter, loss_lan.item(), loss_regid.item(), loss_regexp.item())
            if iter % 1000 == 0 and iter >= 1000:
                for param_group in optimizer_idexp.param_groups:
                    param_group["lr"] *= 0.2
                for param_group in optimizer_frame.param_groups:
                    param_group["lr"] *= 0.2

        print(f'    loss值：{loss_lan.item()},均值:{torch.mean(trans[:, 2]).item()}')
        return light_para, tex_para, euler_angle, trans, exp_para, id_para, focal_length

    def fit_光栅(self, light_para, num_frames, img_paths, lms, tex_para, euler_angle, trans, exp_para, id_para,
                 model_3dmm, focal_length, cxy):
        print(f'[INFO] 光栅...')
        sel_ids = np.arange(0, num_frames, int(num_frames / batch_size))[:batch_size]
        print(f'    生成的sel_ids为：{sel_ids}')
        imgs = []
        for sel_id in sel_ids:
            imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
        print(f'    将图片[{str(len(imgs))}]张读取出...')
        imgs = np.stack(imgs)
        sel_imgs = torch.as_tensor(imgs).cuda()
        sel_lms = lms[sel_ids]
        sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
        set_requires_grad([sel_light])

        optimizer_tl = torch.optim.Adam([tex_para, sel_light], lr=0.1)
        optimizer_id_frame = torch.optim.Adam([euler_angle, trans, exp_para, id_para], lr=0.01)

        print('     循环71次...')
        for iter in range(71):
            sel_exp_para, sel_euler, sel_trans = (
                exp_para[sel_ids],
                euler_angle[sel_ids],
                trans[sel_ids],
            )
            sel_id_para = id_para.expand(batch_size, -1)
            geometry = model_3dmm.get_3dlandmarks(
                sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy
            )
            proj_geo = forward_transform(geometry, sel_euler, sel_trans, focal_length, cxy)

            loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
            loss_regid = torch.mean(id_para * id_para)
            loss_regexp = torch.mean(sel_exp_para * sel_exp_para)

            sel_tex_para = tex_para.expand(batch_size, -1)
            sel_texture = model_3dmm.forward_tex(sel_tex_para)
            geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            rott_geo = forward_rott(geometry, sel_euler, sel_trans)
            render_imgs = self.renderer(
                rott_geo.to(device_render),
                sel_texture.to(device_render),
                sel_light.to(device_render),
            )
            render_imgs = render_imgs.to(device_default)

            mask = (render_imgs[:, :, :, 3]).detach() > 0.0
            render_proj = sel_imgs.clone()
            render_proj[mask] = render_imgs[mask][..., :3].byte()
            loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)

            if iter > 50:
                loss = loss_col + loss_lan * 0.05 + loss_regid * 1.0 + loss_regexp * 0.8
            else:
                loss = loss_col + loss_lan * 3 + loss_regid * 2.0 + loss_regexp * 1.0

            optimizer_tl.zero_grad()
            optimizer_id_frame.zero_grad()
            loss.backward()

            optimizer_tl.step()
            optimizer_id_frame.step()

            if iter % 50 == 0 and iter > 0:
                for param_group in optimizer_id_frame.param_groups:
                    param_group["lr"] *= 0.2
                for param_group in optimizer_tl.param_groups:
                    param_group["lr"] *= 0.2
            render_imgs = None
            geometry = None
            rott_geo = None
            del render_imgs
            del geometry
            del rott_geo
            gc.collect()
            print(iter, loss_col.item(), loss_lan.item(), loss_regid.item(), loss_regexp.item())

        light_mean = torch.mean(sel_light, 0).unsqueeze(0).repeat(num_frames, 1)
        light_para.data = light_mean

        exp_para = exp_para.detach()
        euler_angle = euler_angle.detach()
        trans = trans.detach()
        light_para = light_para.detach()


        return exp_para, euler_angle, trans, light_para

    def fit_精细拟合(self, num_frames, img_paths, lms, exp_para, exp_dim, euler_angle, trans, light_para, id_para,
                     tex_para, model_3dmm, focal_length, cxy, ori_img_path):
        print(f'[INFO] 精细框架拟合...')
        iterCount = int((num_frames - 1) / batch_size + 1)
        print(f'循环{iterCount}次')
        lastIterCount = 0
        # 从存储文件中读取之前存储的运行数据，可以让之前断掉的执行继续执行
        track_iter_params_file = os.path.join(os.path.dirname(ori_img_path), "track_iter_params.pt")
        if os.path.exists(track_iter_params_file):
            lastD = torch.load(track_iter_params_file)
            if lastD:
                lastIterCount = lastD['lastIterCount']
                if lastIterCount and str(lastIterCount).isdigit() and lastIterCount < iterCount:
                    print('从上次断掉的循环中恢复出运行参数，继续执行...')
                    print(f'已循环{lastIterCount}/{iterCount}次')
                    lastIterCount = lastD['lastIterCount']
                    lms = lastD['lms']
                    exp_para = lastD['exp_para']
                    exp_dim = lastD['exp_dim']
                    euler_angle = lastD['euler_angle']
                    trans = lastD['trans']
                    light_para = lastD['light_para']
                    id_para = lastD['id_para']
                    tex_para = lastD['tex_para']
                    focal_length = lastD['focal_length']
                    cxy = lastD['cxy']

        return self.fit_精细拟合_private_iter(iterCount=iterCount, lastIterCount=lastIterCount, num_frames=num_frames,
                                              img_paths=img_paths, lms=lms,
                                              exp_para=exp_para,
                                              exp_dim=exp_dim, euler_angle=euler_angle, trans=trans,
                                              light_para=light_para,
                                              id_para=id_para,
                                              tex_para=tex_para, model_3dmm=model_3dmm, focal_length=focal_length,
                                              cxy=cxy,
                                              ori_img_path=ori_img_path
                                              )

    def fit_精细拟合_private_iter(self, iterCount, lastIterCount, num_frames, img_paths, lms, exp_para, exp_dim,
                                  euler_angle, trans, light_para, id_para,
                                  tex_para, model_3dmm, focal_length, cxy, ori_img_path):
        if not lastIterCount:
            lastIterCount = 0
        id_para.to(device_render)
        # exp_para.to(device_render)
        tex_para.to(device_render)
        cxy.to(device_render)
        focal_length.to(device_render)
        trans.to(device_render)
        euler_angle.to(device_render)

        for i in range(lastIterCount, iterCount):
            lastIterCount = i
            if (i + 1) * batch_size > num_frames:
                start_n = num_frames - batch_size
                sel_ids = np.arange(num_frames - batch_size, num_frames)
            else:
                start_n = i * batch_size
                sel_ids = np.arange(i * batch_size, i * batch_size + batch_size)

            imgs = []
            for sel_id in sel_ids:
                imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
            imgs = np.stack(imgs)
            print(f'第[{i + 1}/{iterCount}]次读取图片{str(len(imgs))}张...')
            sel_imgs = torch.as_tensor(imgs).cuda()
            sel_lms = lms[sel_ids]

            sel_exp_para = exp_para.new_zeros((batch_size, exp_dim), requires_grad=True)
            sel_exp_para.data = exp_para[sel_ids].clone()
            sel_euler = euler_angle.new_zeros((batch_size, 3), requires_grad=True)
            sel_euler.data = euler_angle[sel_ids].clone()
            sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
            sel_trans.data = trans[sel_ids].clone()
            sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
            sel_light.data = light_para[sel_ids].clone()

            set_requires_grad([sel_exp_para, sel_euler, sel_trans, sel_light])
            if lastIterCount > 0:
                exp_para = exp_para.clone()
                euler_angle = euler_angle.clone()
                trans = trans.clone()
                light_para = light_para.clone()

            optimizer_cur_batch = torch.optim.Adam(
                [sel_exp_para, sel_euler, sel_trans, sel_light], lr=0.005
            )

            # if lastIterCount > 0:
            #     exp_para = exp_para.clone()
            #     euler_angle = euler_angle.clone()
            #     trans = trans.clone()
            #     light_para = light_para.clone()


            sel_id_para = id_para.expand(batch_size, -1).detach()
            sel_tex_para = tex_para.expand(batch_size, -1).detach()
            sel_id_para = sel_id_para.cuda()
            sel_tex_para = sel_tex_para.cuda()

            pre_num = 5

            if i > 0:
                pre_ids = np.arange(start_n - pre_num, start_n)

            lossTime = 50
            lossBar = Bar(f'     子循环:', max=lossTime)
            for iter in range(lossTime):
                geometry = model_3dmm.get_3dlandmarks(
                    sel_id_para,
                    sel_exp_para,
                    sel_euler,
                    sel_trans,
                    focal_length,
                    cxy
                )
                proj_geo = forward_transform(geometry, sel_euler, sel_trans, focal_length, cxy)
                loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
                loss_regexp = torch.mean(sel_exp_para * sel_exp_para)

                # sel_geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
                sel_texture = model_3dmm.forward_tex(sel_tex_para)
                geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
                rott_geo = forward_rott(geometry, sel_euler, sel_trans)
                render_imgs = self.renderer(
                    rott_geo.to(device_render),
                    sel_texture.to(device_render),
                    sel_light.to(device_render),
                )
                render_imgs = render_imgs.to(device_default)
                # print(f'         第{iter + 1}次循环，得到渲染图片...')

                mask = (render_imgs[:, :, :, 3]).detach() > 0.0

                loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)

                if i > 0:
                    geometry_lap = model_3dmm.forward_geo_sub(
                        id_para.expand(batch_size + pre_num, -1).detach(),
                        torch.cat((exp_para[pre_ids].detach(), sel_exp_para)),
                        model_3dmm.rigid_ids,
                    )
                    rott_geo_lap = forward_rott(
                        geometry_lap,
                        torch.cat((euler_angle[pre_ids].detach(), sel_euler)),
                        torch.cat((trans[pre_ids].detach(), sel_trans)),
                    )
                    loss_lap = cal_lap_loss(
                        [rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)], [1.0]
                    )
                else:
                    geometry_lap = model_3dmm.forward_geo_sub(
                        id_para.expand(batch_size, -1).detach(),
                        sel_exp_para,
                        model_3dmm.rigid_ids,
                    )
                    rott_geo_lap = forward_rott(geometry_lap, sel_euler, sel_trans)
                    loss_lap = cal_lap_loss(
                        [rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)], [1.0]
                    )

                if iter > 30:
                    loss = loss_col * 0.5 + loss_lan * 1.5 + loss_lap * 100000 + loss_regexp * 1.0
                else:
                    loss = loss_col * 0.5 + loss_lan * 8 + loss_lap * 100000 + loss_regexp * 1.0

                optimizer_cur_batch.zero_grad()
                loss.backward()
                optimizer_cur_batch.step()

                if iter + 1 != lossTime:
                    render_imgs = None
                    del render_imgs
                geometry = None
                rott_geo = None
                del geometry
                del rott_geo
                # 设置进度条完成一格
                lossBar.next()

                gc.collect()

                # if iter % 10 == 0:
                #     print(
                #         i,
                #         iter,
                #         loss_col.item(),
                #         loss_lan.item(),
                #         loss_lap.item(),
                #         loss_regexp.item(),
                #     )

            # 清空进度条
            lossBar.finish()

            render_proj = sel_imgs.clone()
            render_proj[mask] = render_imgs[mask][..., :3].byte()

            exp_para[sel_ids] = sel_exp_para.clone()
            euler_angle[sel_ids] = sel_euler.clone()
            trans[sel_ids] = sel_trans.clone()
            light_para[sel_ids] = sel_light.clone()

            # 每执行2次就存储一份模型文件，避免因为长时间执行服务连接断开而模型文件无法保存的问题
            if i > 0 and i % 2 == 0:
                torch.save(
                    {
                        "id": id_para.detach().cpu(),
                        "exp": exp_para.detach().cpu(),
                        "euler": euler_angle.detach().cpu(),
                        "trans": trans.detach().cpu(),
                        "focal": focal_length.detach().cpu(),
                    },
                    os.path.join(os.path.dirname(ori_img_path), "track_params_tmp.pt"),
                )
            # 每执行1次就存储1次执行参数信息，方便如果执行断开的话，下次继续执行
            torch.save({
                "lastIterCount": lastIterCount,
                "lms": lms,
                "exp_para": exp_para,
                "exp_dim": exp_dim,
                "euler_angle": euler_angle,
                "trans": trans,
                "light_para": light_para,
                "id_para": id_para,
                "tex_para": tex_para,
                "focal_length": focal_length,
                "cxy": cxy
            }, os.path.join(os.path.dirname(ori_img_path), "track_iter_params.pt"))

            gc.collect()
        #
        # 执行完成，返回
        return id_para, exp_para, euler_angle, trans, light_para
