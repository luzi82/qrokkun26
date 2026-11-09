extends Node2D
## Qrokkun — title / play / game-over. One endless wave.

enum State { TITLE, PLAYING, GAMEOVER }

const BULLET_SCENE := preload("res://scenes/bullet.tscn")
const TEX_TOKKUN := preload("res://assets/tokkun.png")

const COL_GOLD := Color("ffcc44")
const COL_CREAM := Color("fff4d2")
const COL_MUTED := Color("c4b89a")
const COL_TIME := Color("ffe566")
const COL_RANK := Color("7ee8ff")
const COL_PROMPT := Color("ffd27a")

@onready var player: Area2D = $Player
@onready var bullets: Node2D = $Bullets
@onready var ui: CanvasLayer = $UI

var state: int = State.TITLE
var elapsed: float = 0.0
var spawn_acc: float = 0.0
var rng := RandomNumberGenerator.new()
var gv = preload("res://scripts/gv20.gd").new()
var monkey: Node = null
var monkey_mode: bool = false

var hud_time: PixelText
var title_name: PixelText
var title_tribute: PixelText
var title_credit: PixelText
var title_prompt: PixelText
var title_best: PixelText
var tokkun_sprite: Sprite2D
var dim: ColorRect
var over_time: PixelText
var over_rank: PixelText
var over_best: PixelText
var over_prompt: PixelText
var flash: ColorRect
var flash_t: float = 0.0


func _ready() -> void:
	monkey_mode = OS.get_environment("QROKKUN_MONKEY") == "1"
	if monkey_mode:
		var seed := int(OS.get_environment("QROKKUN_SEED"))
		if OS.get_environment("QROKKUN_SEED") == "":
			seed = 42
		rng.seed = seed
	else:
		rng.randomize()
	var collision := Node.new()
	collision.set_script(preload("res://scripts/gv20_collision.gd"))
	collision.game = self
	add_child(collision)
	player.monitoring = false
	player.died.connect(_on_player_died)
	player.freeze()
	player.place_center()
	_build_ui()
	_show_title()
	if monkey_mode:
		call_deferred("_monkey_boot")


func _monkey_boot() -> void:
	var seed := int(OS.get_environment("QROKKUN_SEED"))
	if OS.get_environment("QROKKUN_SEED") == "":
		seed = 42
	var out_path := OS.get_environment("QROKKUN_OUT")
	if out_path == "":
		out_path = "user://monkey.jsonl"
	var frames := int(OS.get_environment("QROKKUN_FRAMES"))
	if frames <= 0:
		frames = 300
	monkey = Node.new()
	monkey.set_script(load("res://scripts/monkey_recorder.gd"))
	add_child(monkey)
	monkey.setup(self, seed, out_path, frames)
	_start_run()
	monkey.begin()


func _build_ui() -> void:
	dim = ColorRect.new()
	dim.color = Color(0.06, 0.03, 0.1, 0.58)
	dim.size = Vector2(320, 240)
	dim.mouse_filter = Control.MOUSE_FILTER_IGNORE
	dim.visible = false
	ui.add_child(dim)

	flash = ColorRect.new()
	flash.color = Color(1.0, 0.25, 0.35, 0.0)
	flash.size = Vector2(320, 240)
	flash.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_child(flash)

	hud_time = _px("0.0", Vector2(160, 5), 1, COL_TIME, 1)

	title_name = _px("QROKKUN", Vector2(160, 46), 3, COL_GOLD, 1)
	tokkun_sprite = Sprite2D.new()
	tokkun_sprite.texture = TEX_TOKKUN
	tokkun_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	tokkun_sprite.position = Vector2(118, 88)
	tokkun_sprite.centered = true
	ui.add_child(tokkun_sprite)
	title_tribute = _px("TRIBUTE", Vector2(178, 84), 1, COL_GOLD, 0)
	title_credit = _px("MADE BY GROK BOT", Vector2(160, 108), 1, COL_MUTED, 1)
	title_prompt = _px("PRESS Z / SPACE", Vector2(160, 150), 1, COL_PROMPT, 1)
	title_best = _px("", Vector2(160, 200), 1, COL_CREAM, 1)

	over_time = _px("", Vector2(160, 64), 2, COL_TIME, 1)
	over_rank = _px("", Vector2(160, 92), 2, COL_RANK, 1)
	over_best = _px("", Vector2(160, 124), 1, COL_MUTED, 1)
	over_prompt = _px("PRESS Z / SPACE", Vector2(160, 164), 1, COL_PROMPT, 1)


func _px(text: String, pos: Vector2, sc: int, col: Color, align: int) -> PixelText:
	var n := PixelText.new()
	n.text = text
	n.px_scale = sc
	n.color = col
	n.align = align
	n.position = pos
	ui.add_child(n)
	return n


func _physics_process(delta: float) -> void:
	if flash_t > 0.0:
		flash_t = maxf(0.0, flash_t - delta)
		flash.color.a = flash_t / 0.18 * 0.45

	match state:
		State.TITLE:
			if not monkey_mode and title_prompt != null:
				title_prompt.visible = int(Time.get_ticks_msec() / 420) % 2 == 0
			if (not monkey_mode) and Input.is_action_just_pressed("start"):
				_start_run()
		State.PLAYING:
			elapsed += delta
			gv.frame += 1
			if hud_time != null:
				hud_time.text = "%.1f" % elapsed
			_spawn_step(delta)
		State.GAMEOVER:
			if not monkey_mode and over_prompt != null:
				over_prompt.visible = int(Time.get_ticks_msec() / 420) % 2 == 0
			if (not monkey_mode) and Input.is_action_just_pressed("start"):
				_start_run()


func _start_run() -> void:
	flash_t = 0.0
	flash.color.a = 0.0
	for c in bullets.get_children():
		bullets.remove_child(c)
		c.queue_free()
	elapsed = 0.0
	spawn_acc = -0.7
	player.place_center()
	player.arm()
	state = State.PLAYING
	_show_play()
	gv.reset(self)
	gv.spawn_step(self)


func _on_player_died() -> void:
	if state != State.PLAYING:
		return
	state = State.GAMEOVER
	# Keep impact-tick velocity observable; locked player stops next tick.
	player.locked = true
	for c in bullets.get_children():
		c.set_physics_process(false)
	var rank := GameState.record(elapsed)
	over_time.text = "%.1f" % elapsed
	over_rank.text = rank
	over_best.text = "BEST %.1f" % GameState.best_time
	flash_t = 0.18
	_show_gameover()


func _spawn_step(_delta: float) -> void:
	gv.spawn_step(self)


func _show_title() -> void:
	hud_time.visible = false
	title_name.visible = true
	title_tribute.visible = true
	title_credit.visible = true
	title_prompt.visible = true
	tokkun_sprite.visible = true
	if GameState.best_time > 0.0:
		title_best.text = "BEST %.1f  %s" % [GameState.best_time, Ranks.title_for(GameState.best_time)]
		title_best.visible = true
	else:
		title_best.visible = false
	dim.visible = false
	over_time.visible = false
	over_rank.visible = false
	over_best.visible = false
	over_prompt.visible = false
	player.visible = true


func _show_play() -> void:
	hud_time.visible = true
	hud_time.text = "0.0"
	title_name.visible = false
	title_tribute.visible = false
	title_credit.visible = false
	title_prompt.visible = false
	title_best.visible = false
	tokkun_sprite.visible = false
	dim.visible = false
	over_time.visible = false
	over_rank.visible = false
	over_best.visible = false
	over_prompt.visible = false


func _show_gameover() -> void:
	hud_time.visible = true
	title_name.visible = false
	title_tribute.visible = false
	title_credit.visible = false
	title_prompt.visible = false
	title_best.visible = false
	tokkun_sprite.visible = false
	dim.visible = true
	over_time.visible = true
	over_rank.visible = true
	over_best.visible = true
	over_prompt.visible = true
