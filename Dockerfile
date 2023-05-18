FROM ros:galactic-ros-base

ENV HOME=/root

RUN apt -y update && apt -y upgrade

RUN apt install -y python3-pip git bash-completion curl python3-tk openssh-client

RUN pip3 install numpy==1.17.4 matplotlib==3.1.2 scipy==1.10.1 sympy==1.12

RUN curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash && \
   echo source ~/.git-completion.bash >> ~/.bashrc

WORKDIR $HOME/scripts

RUN git config --system user.name "Joshua Ra'anan" && \
    git config --system user.email "joshua.raanan@gmail.com" && \
    # This is unsafe outside a docker  container
    git config --system --add safe.directory /root/scripts

CMD ["tail", "-f", "/dev/null"]
