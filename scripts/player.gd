extends Area2D
## Grok Bot mascot (white body, black eyes). Snap-stop. Hitbox slightly smaller than sprite.

signal died

const MAX_SPEED := 102.0
const ACCEL := 980.0
const MARGIN := 6.0

const TEX_IDLE := preload("res://assets/player.png")
const TEX_N := preload("res://assets/player_n.png")
const TEX_NE := preload("res://assets/player_ne.png")
const TEX_E := preload("res://assets/player_e.png")
const TEX_SE := preload("res://assets/player_se.png")
const TEX_S := preload("res://assets/player_s.png")
const TEX_SW := preload("res://assets/player_sw.png")
const TEX_W := preload("res://assets/player_w.png")
const TEX_NW := preload("res://assets/player_nw.png")

var velocity: Vector2 = Vector2.ZERO
var locked: bool = true
var dead: bool = false
var use_force_dir: bool = false
var force_dir: Vector2 = Vector2.ZERO


func _ready() -> void:
	area_entered.connect(_on_area_entered)


func _physics_process(delta: float) -> void:
	if locked or dead:
		velocity = Vector2.ZERO
		return
	var dir := Vector2.ZERO
	if use_force_dir:
		dir = force_dir
	else:
		if Input.is_action_pressed("move_left"):
			dir.x -= 1.0
		if Input.is_action_pressed("move_right"):
			dir.x += 1.0
		if Input.is_action_pressed("move_up"):
			dir.y -= 1.0
		if Input.is_action_pressed("move_down"):
			dir.y += 1.0
	if dir != Vector2.ZERO:
		dir = dir.normalized()
		velocity = dir * MAX_SPEED
		_set_look(dir)
	else:
		# Snap-stop on key release (no coasting slide).
		velocity = Vector2.ZERO
		_set_look(Vector2.ZERO)
	global_position += velocity * delta
	var field: Rect2 = GameState.FIELD
	global_position.x = clampf(global_position.x, field.position.x + MARGIN, field.end.x - MARGIN)
	global_position.y = clampf(global_position.y, field.position.y + MARGIN, field.end.y - MARGIN)


func _on_area_entered(area: Area2D) -> void:
	if locked or dead:
		return
	if area.is_in_group("bullets"):
		dead = true
		locked = true
		died.emit()


func _set_look(dir: Vector2) -> void:
	var spr: Sprite2D = $Sprite2D
	if dir == Vector2.ZERO:
		spr.texture = TEX_IDLE
		return
	# 8-way from input axes (not normalized diagonal length).
	var x := 0
	var y := 0
	if absf(dir.x) > 0.01:
		x = 1 if dir.x > 0.0 else -1
	if absf(dir.y) > 0.01:
		y = 1 if dir.y > 0.0 else -1
	match Vector2i(x, y):
		Vector2i(0, -1):
			spr.texture = TEX_N
		Vector2i(1, -1):
			spr.texture = TEX_NE
		Vector2i(1, 0):
			spr.texture = TEX_E
		Vector2i(1, 1):
			spr.texture = TEX_SE
		Vector2i(0, 1):
			spr.texture = TEX_S
		Vector2i(-1, 1):
			spr.texture = TEX_SW
		Vector2i(-1, 0):
			spr.texture = TEX_W
		Vector2i(-1, -1):
			spr.texture = TEX_NW
		_:
			spr.texture = TEX_IDLE


func arm() -> void:
	dead = false
	locked = false
	modulate = Color.WHITE
	velocity = Vector2.ZERO
	visible = true
	_set_look(Vector2.ZERO)


func freeze() -> void:
	locked = true
	velocity = Vector2.ZERO


func place_center() -> void:
	var field: Rect2 = GameState.FIELD
	global_position = field.position + field.size * 0.5
	velocity = Vector2.ZERO
