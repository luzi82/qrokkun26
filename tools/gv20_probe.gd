extends SceneTree
var game
var tick := 0
var out: FileAccess
var limit := 7200

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	game = load("res://scenes/main.tscn").instantiate()
	root.add_child(game)
	game.gv.capture_events = OS.get_environment("GV20_MEMORY") != "1"
	var replay_path := OS.get_environment("GV20_REPLAY")
	if replay_path != "":
		game.gv.replay = JSON.parse_string(FileAccess.get_file_as_string(replay_path))
	if OS.get_environment("GV20_INPUT") == "1":
		var driver := Node.new()
		driver.set_script(load("res://tools/gv20_input.gd"))
		driver.game = game
		root.add_child(driver)
	else:
		game._start_run()
	if OS.get_environment("GV20_LONG") == "1":
		game.get_child(game.get_child_count() - 1).set_physics_process(false)
	var boundary := OS.get_environment("GV20_BOUNDARY")
	if boundary != "":
		for b in game.bullets.get_children():
			b.position = game.player.position + Vector2(float(boundary), 0)
			b.velocity = Vector2.ZERO
	if OS.get_environment("GV20_CAPTURE") != "":
		await RenderingServer.frame_post_draw
		var image := root.get_texture().get_image()
		image.save_png(OS.get_environment("GV20_CAPTURE"))
		quit()
		return
	if OS.get_environment("GV20_INPUT") != "1" and game.bullets.get_child_count() != 1:
		push_error("Gv20 reset must spawn exactly one frame-zero bullet")
		quit(1)
		return
	out = FileAccess.open(OS.get_environment("GV20_TRACE"), FileAccess.WRITE)
	if out == null:
		print("RESET GREEN")
		quit()
		return
	var observer := Node.new()
	observer.set_script(load("res://tools/gv20_trace.gd"))
	observer.game = game
	observer.out = out
	root.add_child(observer)
