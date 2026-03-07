import melee

# 1. Setup: Point to your Dolphin executable and ISO
# If running in a notebook, ensure XVFB is running as discussed before!
console = melee.Console(path="path/to/slippi-dolphin")
controller = melee.Controller(console=console, port=1)

# 2. Start the emulator
console.run()

# 3. The 60 FPS Loop
while True:
    # console.step() blocks until the NEXT frame is ready from Dolphin.
    # This automatically syncs your Python script to the game's 60fps clock.
    gamestate = console.step()
    
    if gamestate is None:
        continue

    # 4. Extract Frame Data
    # Access player 1 (Port 1) data
    p1 = gamestate.player[1]
    
    # Example: Print position and action every frame
    print(f"Frame: {gamestate.frame} | Pos: ({p1.x:.2f}, {p1.y:.2f}) | Action: {p1.action}")

    # 5. Send an Input (Crucial for RL)
    # The AI must decide an action here.
    controller.tilt_stick(melee.Stick.MAIN, 0.5, 0.5) # Example: Walk right
    
    # If the game ends or you want to stop:
    if gamestate.menu_state == melee.Menu.POSTGAME:
        break