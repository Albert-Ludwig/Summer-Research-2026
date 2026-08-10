WIRED CONNECTION OVER ETHERNET

1. Turn on jackal and connect controller

ON JACKAL
2. Set the IP address of your laptops wired connection 

go to settings -> network -> wired -> gear icon -> IPv4

Switch to Manual
address: 192.168.131.101
netmask: 255.255.255.0

3. SSH into the robot, if doing over ethernet, connect ethernet cable to laptop.

ssh administrator@192.168.131.1

4. Once SSH'd into the robot, check that all the topics and nodes are available

ON LAPTOP
1. Download the fastdds_robot_wired.xml and ros_ethernet.env file, store them in the same location.
2. In your terminal, go to the folder that has these files and run source ros_ethernet.env, you should see your laptops IP and ethernets IP connected.



WIRELESS CONNECTION OVER NETWORK
ROBOHUB IP ADDRESS: 129.97.71.36

1. Turn on jackal and connect controller

ON JACKAL
1. Ensure the jackal is connected to your network, if it is, the green wifi symbol on the back of the jackal should be green.

IF IT IS NOT CONNECTED:
1. SSH into the jackal over ethernet
2. cd /etc/netplan
3. sudo nano wifi-60.yaml

In this yaml file, ONLY EDIT the bottom fields where you add in your access point name and wifi password.

CTRL+O to write and enter
CTRL+X to exit

4. Apply the changes
sudo netplan apply (or restart the jackal)

5. After applying the changes, get the robots new ip address, do this over ethernet at first.

ip addr show - note down the ip address being shown under wlp2s0

6. Exit SSH over ethernet, connect your laptop to the same network as the jackal and ping this new ip address
ping <ip address>

7. If ping successful, SSH into the robot using this new IP address
ssh administrator@<ip address>

ON LAPTOP
1. Download fastdds_robot_template.xml and ros_robot.env file, store them in the same location. 
MAKE SURE TO UPDATE THE LINE 31 WITH THE PATH TO WHERE YOU HAVE YOUR FILES STORED

2. source ros_robot.env <ip address of the robot>

3. Now you should be connected wirelessly!
