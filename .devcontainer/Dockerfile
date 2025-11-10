FROM python:3

ARG USERNAME=yc-dlan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update \
    && apt-get install -y  sudo vim nautilus 

# 创建用户组和用户
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # 配置环境变量
    && echo "$USERNAME:'" | chpasswd \
    # 允许用户使用 sudo
    && usermod -aG sudo $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    # 使新用户的 `.bashrc` 文件生效
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

USER ${USERNAME}

# 安装fish插件（切换到普通用户）
RUN curl -L https://github.com/oh-my-fish/oh-my-fish/raw/master/bin/install | sudo tee install_omf > /dev/null || true && \
    fish install_omf --noninteractive || true && \
    fish -c "omf install bass" || true && \
    sudo rm install_omf || true

# 复制fish配置
COPY packages/fish /home/${USERNAME}/.config/fish

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
#安装机器学习相关依赖
RUN pip install scikit-learn xgboost

#安葬web服务框架
RUN pip install fastapi uvicorn gunicorn

#安装部分计算库
RUN pip install pandas numpy joblib redis kafka-python

#安装数据库相关
RUN pip install mysqlclient psycopg2-binary sqlalchemy

#安装http请求库，及部分实用库
RUN pip install requests pydantic python-multipart

USER root

RUN echo "export PATH=\$HOME/.local/bin:\$PATH" >> /home/$USERNAME/.bashrc \
    && echo "set -gx PATH \$HOME/.local/bin \$PATH" >> /home/$USERNAME/.config/fish/config.fish

ENV PYTHONPATH = /home/$USERNAME/py_ws
ENV PYTHONPATH = ~/py_ws
ENV PYTHONPATH = /home/$USERNAME/py_ws
