# paper5 实验启动

在远程 A800 服务器的同一个 shell 中依次执行：

1. 进入仓库：`cd ~/whr/paper5/repo`
2. 激活环境：`conda activate autoresearch_paper5`
3. 选择 GPU：`source paper5/scripts/select_gpu.sh`
4. 运行已批准的实验。

脚本通过 `nvidia-smi` 选择当前显存占用最低的 GPU，并设置 `CUDA_VISIBLE_DEVICES`。该变量只作用于当前 shell 及其启动的子进程；选择失败时应停止启动实验。显存读数是选择时的快照，并不预留 GPU。

H1 后续实验输出应保存在 `paper5/results/`。
