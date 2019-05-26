# Reinforcement Learning for pathfinding in Matlab

This Matlab framework includes a gridworld and a reinforcement learning approach to learn how to navigate in this gridworld.

## Description

The generalimformation about the aim of this project is given in [`Gridworls_Coursework.pdf`](Gridworlds_Coursework1.pdf). The gridworld dynamics is provided by the [`PersonalisedGridWorld.p`](PersonalisedGridWorld.p) file, it is non readable and should not be changed (just to be used in the code to generate the gridworld). The reinforcement learning approach was implemented in [`Coursework1.m`](Coursework1.m) file. The results of the implementation is given in [`CourseworkReport.pdf`](CourseworkReport.pdf)

## Dependencies

- Matlab

This project does't require dependencies to any other external software or libraries.

## Guide

The gridworld dynamics was callded by the main coursework script. All of the required functions are also in the main script. To run the program, simply run the [`Coursework1.m`](Coursework1.m) script on Matlab. 

You can change the learning parameters of the program (like the discount constant for the value function or probability constant for the transition matrix) and see how that affects learning. The script will plot the learning curve at the end.
