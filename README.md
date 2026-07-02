# HelloBalls: Autonomous Tennis Court Assistant  - Your AI ball boy

**HelloBalls** is an autonomous tennis court assistant designed to partially replace the role of a tennis ball boy.  
It supports players by taking over repetitive on-court tasks such as ball collection, ball delivery, player following, and gesture-based interaction.

Unlike a traditional tennis ball machine, helloBalls is not only focused on serving balls for training. Instead, it acts as a robotic assistant that helps maintain the practice flow by reducing interruptions caused by picking up balls or manually controlling equipment.

---
## Overview

During tennis practice, players often need to stop frequently to pick up balls, reposition equipment, or ask another person to assist with ball delivery. These repetitive tasks interrupt the rhythm of practice and make solo practice less efficient.

HelloBalls addresses this problem by combining computer vision, robotic control, and mechanical design to create an intelligent on-court assistant that can:

- Recognize user gestures to understand player commands
- Follow the player on court
- Collect tennis balls automatically
- Deliver or feed balls when needed
- Coordinate perception, control, and mechanical actions through a system-level state machine

The goal of this project is to explore how AI and robotics can support tennis players by replacing part of the traditional ball-boy role.

---

## Key Features

### Vision Module

HelloBalls uses a YOLO11-based vision module to support real-time perception on the tennis court.  
The vision module includes three main functions: gesture recognition, player following, and tennis ball recognition.

#### Gesture Recognition

The gesture recognition function allows the player to control HelloBalls through simple hand gestures.  
This enables hands-free interaction, so the player can give commands while still holding a racket.

Supported gesture commands include:

| Gesture | Robot Command |
|---|---|
| One | Serve one ball |
| OK | Pick up balls |
| Palm | Switch target |
| Fist | Stop |

#### Player Following

The system uses YOLO11 to detect and track the player on the tennis court.  
This allows HelloBalls to follow the player and stay within an appropriate assisting range without requiring constant manual control.

#### Tennis Ball Recognition

The system uses YOLO11 to recognize tennis balls on the court.  
The detected ball positions are used to support automatic ball collection and help the robot locate balls more efficiently.

### Automatic Ball Collection

A mechanical ball-picking structure is designed to collect tennis balls from the court.  
This reduces the need for players to repeatedly stop and pick up balls manually.

### Ball Delivery / Feeding

The robot supports controlled ball delivery, allowing it to act more like an on-court assistant rather than a traditional tennis ball machine.

### State Machine Control

A system-level state machine manages different operating modes, including idle, player following, ball recognition, ball collection, and ball delivery.  
This makes the robot behavior more stable, predictable, and easier to debug.

Poster: [HelloBalls-Poster-CMYK.pdf](https://github.com/user-attachments/files/27423382/HelloBalls-Poster-CMYK.pdf)

Pitch Video: https://github.com/user-attachments/assets/8cc95ef7-b23d-4885-80cf-73ba0a497ce6

Poster:
<img width="5264" height="7450" alt="HelloBalls-Poster-CMYK-images-1" src="https://github.com/user-attachments/assets/122d1883-a26f-4292-a723-0c315ef401cc" />
<img width="5264" height="7450" alt="HelloBalls-Poster-CMYK-images-0" src="https://github.com/user-attachments/assets/36ad99ec-0db4-4734-bffd-e017a2d7edba" />






**1. Project Idea**

This project aims to design and implement a smart tennis machine that can automatically collect balls, store them, and serve them back to players on demand. Unlike traditional tennis machines that only serve balls, our system introduces computer vision–based interaction and intelligent ball retrieval, providing a seamless training experience. The motivation comes from addressing a common frustration in tennis practice: wasting time and energy picking up scattered balls.


**2. User Study**

We conducted a small-scale user study involving tennis players, coaches, and recreational users. The study revealed several pain points in current training sessions:
 - Frequent interruptions due to manual ball collection.
 - Limited interactivity in existing ball-serving machines.
 - A desire for more natural interaction methods, such as gestures or voice.
From this, we derived key user requirements:
 - Automated ball collection and serving.
 - Lightweight and portable design.
 - Intuitive control through gestures or voice commands.
 - Reliability and stability during long training sessions.


**3. Background Research on Related Product**

Existing products such as Lobster and Slinger Bag provide effective ball serving but lack the ability to collect balls or integrate intelligent interaction. Some high-end machines offer advanced serving patterns, but they remain purely one-directional (machine to player).
 Our research showed a clear gap in the market: no consumer-level solution combines autonomous ball retrieval, smart serving, and natural human–machine interaction. This guided our system design choices.


** 4. Finalized System Specifications**

System diagram
<img width="1537" height="342" alt="image" src="https://github.com/user-attachments/assets/72f2efc9-aec4-4519-afb5-e04e6cc8ae9a" />




