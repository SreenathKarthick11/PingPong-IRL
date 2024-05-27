import pygame, sys, random

#ball animation function
def ball_animation():
    global ball_speed_x,ball_speed_y,left_score,right_score
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1
    if ball.left <= 0 :
        right_score += 1
    elif ball.right >= screen_width :
        left_score += 1
    if ball.left <= 0 or ball.right >= screen_width:
        ball_restart()
    
    if ball.colliderect(player) or ball.colliderect(opponent):
         ball_speed_x *= -1 
    
# player animation function
def player_animation():
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height

# opponent animation function
def opponent_animation():
    opponent.y += opponent_speed
    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        opponent.bottom = screen_height

def ball_restart():
    global ball_speed_y,ball_speed_x
    ball.center = (screen_width/2,screen_height/2)
    ball_speed_y *= random.choice((1,-1))
    ball_speed_x *= random.choice((1,-1))

# General Setup
pygame.init()
clock=pygame.time.Clock()

# setting up with main window
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption('PingPong')

# Game rectangles
ball = pygame.Rect(screen_width/2 - 15,screen_height/2 -15,30,30)
player = pygame.Rect(screen_width-20,screen_height/2-40,10,80)
opponent = pygame.Rect(10,screen_height/2-40,10,80)

bg_color = pygame.Color('grey12')
light_grey = (200,200,200)

ball_speed_x = 7 * random.choice((1,-1))
ball_speed_y = 7 * random.choice((1,-1))
player_speed = 0
opponent_speed = 0
left_score = 0
right_score = 0
Score_font = pygame.font.SysFont('arial', 50)
left_score_text = Score_font.render(f"{left_score}",True,light_grey)
right_score_text = Score_font.render(f"{right_score}",True,light_grey)

while True:
    # handeling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:#closing the window with x 
            pygame.quit()
            sys.exit()
# player controls
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed += 7
            if event.key == pygame.K_UP:
                player_speed -= 7
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 7
            if event.key == pygame.K_UP:
                player_speed += 7
# opponent controls
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                opponent_speed += 7
            if event.key == pygame.K_w:
                opponent_speed -= 7
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_s:
                opponent_speed -= 7
            if event.key == pygame.K_w:
                opponent_speed += 7
    
 
    ball_animation()
    player_animation()
    opponent_animation()

    #visuals
    screen.fill(bg_color)
    
    pygame.draw.rect(screen,light_grey,player)
    left_score_text = Score_font.render(f"Player 1 : {left_score}",True,light_grey)
    right_score_text = Score_font.render(f"Player 2 : {right_score}",True,light_grey)
    screen.blit(left_score_text,(screen_width/4-left_score_text.get_width()/2,20))
    screen.blit(right_score_text,(screen_width*3/4-right_score_text.get_width()/2,20))
    pygame.draw.rect(screen,light_grey,opponent)
    pygame.draw.ellipse(screen,light_grey,ball)
    pygame.draw.aaline(screen,light_grey,(screen_width/2,0),(screen_width/2,screen_height))
    

    # Updating the window
    pygame.display.flip()
    clock.tick(60)