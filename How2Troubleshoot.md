# How to troubleshoot Tiago Dual for this repo

## Main problems 
While I was working on making this repo work with ROS Noetic and Tiago++ I've come across some issues that took me a bit of time to troubleshoot. 
Hopefully this guide can be helpful for anybody coming across the same issues.

## First issue 
While trying to run `roslaunch tiago_dual_pick_place pick_place.launch` you might come across an issue, where **pick_place_server.py** cannot connect to 
**/pickup** and **/place** Action Servers in this case the issues the approach, which worked for me would be: 

1. Specifing needed env variables: 
```bash
export ROS_MASTER_URI=<TIAGO\'s IP:PORT>
export ROS_IP=<Result of `hostname -I`>
```
> and then running the launch file

2. Checking if **/move_group** node is launched:
```bash
	rosnode list | grep move_group
```
> if no result is returned try restarting Tiago  or manually starting the node for example:
```bash
	roslaunch tiago_dual_moveit_config move_group.launch
```
> remember to run it in a separate terminal window!


### These few steps did wonders for me!