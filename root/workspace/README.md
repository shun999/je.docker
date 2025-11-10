# 共同研究の深層学習コードをまとめたリポジトリ
## 環境
* Python 3.8.10  
* tensorflowのイメージを利用  
* 詳細は"requirements.txt"参照  

## 作成方法（初回）
docker pull tensorflow/tensorflow:latest-gpu #latest-gpuだとうまく動作しないかも  
docker pull tensorflow/tensorflow:2.13.0-gpu #こっちだと確実に動作する（Python3.8.10になるが）  

mkdir workdir && cd workdir(必要なら)  

docker run -it \​  
-v $(pwd):/app \​  
-p 8080:8080 \​  
--gpus=all \​  
--shm-size=256m \​  
--name something \​ #somethingは自分の分かりやすい名前に変える  
tensorflow/tensorflow:latest-gpu bash  

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz​  
tar -xf vscode_cli.tar.gz  

./code tunnel  

## 利用方法（2回目から）
docker start -i something #somethingではなく自分で決めた名前のコンテナを指定する  
./code tunnel  

