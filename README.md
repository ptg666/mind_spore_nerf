# mind_spore_nerf

欢迎大家提出意见帮助我改进它。

Thanks for giving me pertinent advice to improve this project.

将 Tiny-nerf（nerf 的简易实现版）迁移到 mindspore 框架上。

NeRF developed in mindspore

log:12.3

     -运行成功

     -能训练nerf的 mlp

     -test 模块未完成
log:12.15

     -自定义了计算图
     
     -实现了展示运行结果模块
     
     -部署到 ModelArts 使用 GPU 加速训练
     
     -训练速度慢，训练效果不太好

### 已修改的Bug
     
     -1.网络初始参数过小
     
     -2.step_forward()
     
     -3.计算图定义 run_NeRF_lego.py
     
     -4.get_rays() 中 ray_directions的计算方法

### 目前存在的问题
     
     -1.训练结果全黑，找不到原因
# 完了，阳了
     
