import cv2
import numpy as np
import pygame, sys, random

# General Setup
pygame.init()
clock = pygame.time.Clock()

# Setting up the main window
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('PingPong')

# Game rectangles
ball = pygame.Rect(screen_width / 2 - 15, screen_height / 2 - 15, 30, 30)
player = pygame.Rect(screen_width - 20, screen_height / 2 - 40, 10, 80)
opponent = pygame.Rect(10, screen_height / 2 - 40, 10, 80)

bg_color = pygame.Color('grey12')
light_grey = (200, 200, 200)
light_blue  = (173,216,250)
light_green = (152,251,152)

ball_speed_x = 5 * random.choice((1, -1))
ball_speed_y = 5 * random.choice((1, -1))
player_speed = 0
opponent_speed = 0
left_score = 0
right_score = 0
Score_font = pygame.font.SysFont('arial', 50)
left_score_text = Score_font.render(f"{left_score}", True, light_grey)
right_score_text = Score_font.render(f"{right_score}", True, light_grey)

# Ball animation function
def ball_animation():
    global ball_speed_x, ball_speed_y, left_score, right_score
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1
    if ball.left <= 0:
        right_score += 1
        ball_restart()
    elif ball.right >= screen_width:
        left_score += 1
        ball_restart()

    if ball.colliderect(player) or ball.colliderect(opponent):
        ball_speed_x *= -1

# Player animation function
def player_animation():
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height

# Opponent animation function
def opponent_animation():
    opponent.y += opponent_speed
    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        opponent.bottom = screen_height

def ball_restart():
    global ball_speed_y, ball_speed_x
    ball.center = (screen_width / 2, screen_height / 2)
    ball_speed_y *= random.choice((1, -1))
    ball_speed_x *= random.choice((1, -1))

# OpenCV setup
cap = cv2.VideoCapture(0)
template1 = cv2.resize(cv2.imread('assets/temp1.jpg', 0),(0,0),fx=0.5,fy=0.5)
template2 = cv2.resize(cv2.imread('assets/temp2.jpg', 0),(0,0),fx=0.5,fy=0.5)
h1, w1 = template1.shape
h2, w2 = template2.shape
method = cv2.TM_CCOEFF_NORMED

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
    blueframe = cv2.bitwise_and(frame, frame, mask=mask1)

    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([32, 25, 25])
    upper_green = np.array([86, 255, 255])
    mask2 = cv2.inRange(hsv2, lower_green, upper_green)
    greenframe = cv2.bitwise_and(frame, frame, mask=mask2)

    grayframe1 = cv2.cvtColor(blueframe, cv2.COLOR_BGR2GRAY)
    grayframe2 = cv2.cvtColor(greenframe, cv2.COLOR_BGR2GRAY)

    result1 = cv2.matchTemplate(grayframe1, template1, method)
    result2 = cv2.matchTemplate(grayframe2, template2, method)
    _, _, _, max_loc1 = cv2.minMaxLoc(result1) # taking only max location because of methord 2
    _, _, _, max_loc2 = cv2.minMaxLoc(result2)

    bottom_right1 = (max_loc1[0] + w1, max_loc1[1] + h1)
    bottom_right2 = (max_loc2[0] + w2, max_loc2[1] + h2)
    cv2.rectangle(blueframe, max_loc1, bottom_right1,(0,0,255), 5)
    cv2.rectangle(greenframe, max_loc2, bottom_right2, (0,0,255), 5)

    # Handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
    # Update player1 speed based on template matching location
    if max_loc2[1] < 80:
        opponent_speed = -7
    else:
        opponent_speed = 7
    # Update player2 speed based on template matching location
    if max_loc1[1] < 80:
        player_speed = -7
    else:
        player_speed = 7

    ball_animation()
    player_animation()
    opponent_animation()

    # Visuals
    screen.fill(bg_color)
    pygame.draw.rect(screen, light_blue, player)
    left_score_text = Score_font.render(f"Player 1: {left_score}", True, light_grey)
    right_score_text = Score_font.render(f"Player 2: {right_score}", True, light_grey)
    screen.blit(left_score_text, (screen_width / 4 - left_score_text.get_width() / 2, 20))
    screen.blit(right_score_text, (screen_width * 3 / 4 - right_score_text.get_width() / 2, 20))
    pygame.draw.rect(screen, light_green, opponent)
    pygame.draw.ellipse(screen, light_grey, ball)
    pygame.draw.aaline(screen, light_grey, (screen_width / 2, 0), (screen_width / 2, screen_height))

    # Updating the window
    pygame.display.flip()
    clock.tick(30)

    # Display the OpenCV windows for debugging
    cv2.imshow('Match1_BLUEFRAME', blueframe)
    cv2.imshow('Match2_GREENFRAME', greenframe)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
