## Docker

Build and start the docker container:
`$ ./docker_build.bash`

Point docker container to use your monitor:
`$ xhost +local:docker`

Start and enter the docker container:
`$ docker exec -it control_scripts-ros-1 bash`

## iLQR

![iLQR](./images/ilqr.gif)


## Quadratic Program Trajectory

![qp_trajectory](./images/qp_trajectory.png)
