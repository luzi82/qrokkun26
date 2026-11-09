extends Node
var game
var out: FileAccess
var tick := 0
var action := 0
var restarts := 0
var last_frame := -1
const DIRS := [Vector2.ZERO, Vector2.UP, Vector2(1,-1), Vector2.RIGHT, Vector2.ONE, Vector2.DOWN, Vector2(-1,1), Vector2.LEFT, Vector2(-1,-1)]

func _ready() -> void:
	process_physics_priority = 100

func _physics_process(_dt: float) -> void:
	if OS.get_environment("GV20_INPUT") == "1":
		if game.state == game.State.TITLE:
			return
		if game.gv.frame == last_frame:
			return
		last_frame = game.gv.frame
		var direction := Input.get_vector("move_left", "move_right", "move_up", "move_down")
		action = DIRS.find(direction)
	var rows := []
	for b in game.bullets.get_children():
		if not b.is_queued_for_deletion():
			rows.append([b.position.x, b.position.y, b.velocity.x, b.velocity.y, b.get_node("CollisionShape2D").shape.radius])
	out.store_line(JSON.stringify({"action":action, "frame":game.gv.frame, "player":[game.player.position.x,game.player.position.y,game.player.velocity.x,game.player.velocity.y], "dead":game.player.dead, "bullets":rows, "events":game.gv.events, "history_size":game.gv.history.size(), "elapsed":game.elapsed, "playing_ui":game.hud_time.visible and not game.over_time.visible, "gameover_ui":game.over_time.visible and game.dim.visible, "flash_t":game.flash_t, "flash_alpha":game.flash.color.a, "best_time":game.get_node("/root/GameState").best_time, "hit_animation":game.player.hit_anim_t}, "", true, true))
	if OS.get_environment("GV20_MEMORY") != "1":
		game.gv.events.clear()
	if OS.get_environment("GV20_INPUT") == "1" and game.gv.frame == 0 and tick > 0:
		restarts += 1
	tick += 1
	if OS.get_environment("GV20_PERTURB") == "1":
		for i in 19:
			game.gv.player_rng.randf()
	if OS.get_environment("GV20_MOVE") == "1":
		action = (tick / 30) as int % 9
		game.player.use_force_dir = true
		game.player.force_dir = DIRS[action]
	if OS.get_environment("GV20_INPUT") != "1" and game.player.dead and OS.get_environment("GV20_RESTART") == "1" and restarts == 0:
		assert(game.player.hit_anim_t > 0.0)
		assert(game.over_time.visible)
		assert(game.flash_t > 0.0)
		game._start_run()
		assert(game.bullets.get_child_count() == 1)
		assert(game.gv.frame == 0 and not game.player.dead)
		assert(game.hud_time.visible and not game.over_time.visible)
		restarts += 1
		action = 0
		return
	if (game.player.dead and (OS.get_environment("GV20_INPUT") != "1" or restarts > 0)) or tick >= 7200:
		out.flush()
		get_tree().quit()
