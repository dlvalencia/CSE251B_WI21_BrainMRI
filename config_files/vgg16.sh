#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Contrast250
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Contrast025
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Contrast150
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Sharp050
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Sharp250
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Sharp002
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Bright250
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Bright050
python3 main_run_exp.py vgg16_LR0.25e-3_NoRotFlip_Bright025
# 3 https://drive.google.com/file/d/1lH0AexdWUcQs9nvAKUa4pYXN9ZoyBwMM/view?usp=sharing
# fileid="1lH0AexdWUcQs9nvAKUa4pYXN9ZoyBwMM"
# filename="round5-initial.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}


# # https://drive.google.com/file/d/1nvMCYgZhr7Xh05kbC38txvOKe7IFwhuN/view?usp=sharing
# fileid="1nvMCYgZhr7Xh05kbC38txvOKe7IFwhuN"
# filename="id-000000xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# 1 https://drive.google.com/file/d/1Qaq9XPxHRwqCvjIXg_lDxrRUQ92GBpKV/view?usp=sharing
# https://drive.google.com/file/d/1XaAsEupjstjNcLaUWLYg_t9SRwLzWWz3/view?usp=sharing
# fileid="1XaAsEupjstjNcLaUWLYg_t9SRwLzWWz3"
# filename="id-000001xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# 2 https://drive.google.com/file/d/1T67-LczZQrvYGM-e5Qm_d0jf5TCz-V_e/view?usp=sharing
# fileid="1T67-LczZQrvYGM-e5Qm_d0jf5TCz-V_e"
# filename="id-000002xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# 3 https://drive.google.com/file/d/1p1dZLU7cnHTjwcLLuqzt0SFSLjWhQioQ/view?usp=sharing
# fileid="1p1dZLU7cnHTjwcLLuqzt0SFSLjWhQioQ"
# filename="id-000003xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# # 4 https://drive.google.com/file/d/1rC6UpkRHCB1qegueU-vnoGPf_hSjXmbO/view?usp=sharing
# fileid="1rC6UpkRHCB1qegueU-vnoGPf_hSjXmbO"
# filename="id-000004xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

#  # 5 https://drive.google.com/file/d/1ugx5STlknmEgUB6gB9WgyFbbSGdQb-UA/view?usp=sharing
# fileid="1ugx5STlknmEgUB6gB9WgyFbbSGdQb-UA"
# filename="id-000005xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# off1 https://drive.google.com/file/d/1m4hvxgpf6qOZbxOybDhpqIMtbw_EsAKi/view?usp=sharing

# off2 https://drive.google.com/file/d/11chQWAdl4UzoSY3HGubBx_Y3_Nhmupl4/view?usp=sharing
# fileid="1m4hvxgpf6qOZbxOybDhpqIMtbw_EsAKi"
# filename="id-000000xx.tar.gz"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
