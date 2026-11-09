extends RefCounted
## Production Gv20 spawner. Replay supplies random variates, never finished bullets.
var frame := 0
var clock := 0.0
var acc := -0.7
var clock_frame := 0
var origin := -1
var next_frame := 0
var s_rng := RandomNumberGenerator.new()
var field_rng := RandomNumberGenerator.new()
var player_rng := RandomNumberGenerator.new()
var history: Array = []
var events: Array = []
var capture_events := false
var replay: Array = []
var replay_index := 0

func reset(_game) -> void:
	frame = 0
	clock = 0.0
	acc = -0.7
	clock_frame = 0
	origin = -1
	history.clear()
	events.clear()
	replay_index = 0
	var seed_text := OS.get_environment("QROKKUN_SEED")
	var base := int(seed_text) if seed_text != "" else int(Time.get_ticks_usec())
	s_rng.seed = base
	field_rng.seed = base + 104729
	player_rng.seed = base + 209759
	advance_event()

func interval(t: float) -> float:
	if t < 8.0: return 0.48 + (0.32 - 0.48) * (t / 8.0)
	if t < 20.0: return 0.32 + (0.20 - 0.32) * ((t - 8.0) / 12.0)
	if t < 35.0: return 0.20 + (0.12 - 0.20) * ((t - 20.0) / 15.0)
	if t < 55.0: return 0.12 + (0.075 - 0.12) * ((t - 35.0) / 20.0)
	if t < 80.0: return 0.075 + (0.048 - 0.075) * ((t - 55.0) / 25.0)
	return 0.042

func advance_event() -> void:
	while true:
		clock_frame += 1
		clock += 1.0 / 60.0
		acc += 1.0 / 60.0
		var delay := interval(clock)
		if acc >= delay:
			acc -= delay
			if origin < 0: origin = clock_frame
			next_frame = clock_frame - origin
			return

func variates(field: bool) -> Array:
	if not replay.is_empty():
		var event: Array = replay[replay_index]
		replay_index += 1
		assert(int(event[0]) == frame and bool(event[1]) == field)
		# Godot JSON decimal parsing can lose a double ULP; draws are float32.
		return Array(PackedFloat32Array(event.slice(2)))
	if field: return [field_rng.randf_range(0.0, TAU), 0.0, 0.0, 0.0]
	return [s_rng.randf_range(0.0, TAU), s_rng.randfn(0.0, 13.2), s_rng.randfn(0.0, 13.2), s_rng.randf_range(30.0, 200.0)]

func spawn_step(game) -> void:
	while not history.is_empty() and history[0][0] < frame - 60:
		history.pop_front()
	if frame == next_frame:
		var v := variates(false)
		add(game, v, float(v[3]), false)
		history.append([frame, float(v[3])])
		advance_event()
	if frame > 0 and frame % 60 == 0:
		var total := 0.0
		var count := 0
		for entry in history:
			if entry[0] >= frame - 60 and entry[0] < frame:
				total += entry[1]
				count += 1
		if count > 0: add(game, variates(true), total / count, true)

func add(game, v: Array, speed: float, field: bool) -> void:
	var direction := Vector2.from_angle(float(v[0]))
	var distance := minf(160.0 / maxf(absf(direction.x), 1e-30), 114.0 / maxf(absf(direction.y), 1e-30))
	var pos := Vector2(160, 126) + direction * distance
	var delta := Vector2.ZERO if field else Vector2(float(v[1]), float(v[2]))
	var aim: Vector2 = game.player.position + delta - pos
	var velocity := (Vector2.RIGHT if aim.length() < 1e-12 else aim.normalized()) * speed
	var bullet = game.BULLET_SCENE.instantiate()
	game.bullets.add_child(bullet)
	bullet.setup(pos, velocity, 1)
	bullet.first_move_frame = frame + (2 if frame == 0 else 1)
	if capture_events:
		events.append([frame, field, v[0], v[1], v[2], speed])
