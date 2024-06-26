FROM python:3.12-slim

ARG USERNAME=user
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000
ARG PASSWORD=user
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ARG HOME=/home/$USERNAME/work

RUN apt update -y && \
    apt install -y build-essential procps curl file git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR "${HOME}"

COPY ./requirements.txt ${HOME}/requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip && \
pip3 install --no-cache-dir -r ${HOME}/requirements.txt

USER $USERNAME

CMD [ "/bin/bash"]