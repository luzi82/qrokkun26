extends Node
## Synthetic physical key events exercise the production InputMap in headless tests.
var game
var started := false
var restarting := false
var move_key := KEY_RIGHT
var start_key := KEY_ENTER
const KEYS := {"UP": KEY_UP, "W": KEY_W, "RIGHT": KEY_RIGHT, "D": KEY_D,
	"DOWN": KEY_DOWN, "S": KEY_S, "LEFT": KEY_LEFT, "A": KEY_A,
	"ENTER": KEY_ENTER, "KP_ENTER": KEY_KP_ENTER, "SPACE": KEY_SPACE, "Z": KEY_Z}

func _ready() -> void:
	process_physics_priority = -100
	move_key = KEYS.get(OS.get_environment("GV20_MOVE_KEY"), KEY_RIGHT)
	start_key = KEYS.get(OS.get_environment("GV20_START_KEY"), KEY_ENTER)

func key_event(code: int, pressed: bool) -> void:
	var event := InputEventKey.new()
	event.physical_keycode = code
	event.pressed = pressed
	Input.parse_input_event(event)
	Input.flush_buffered_events()

func _physics_process(_dt: float) -> void:
	key_event(start_key, false)
	if not started or (game.state == game.State.GAMEOVER and not restarting):
		# Keep movement held across restart; do not manufacture a second key down.
		if not started:
			key_event(move_key, true)
		key_event(start_key, true)
		started = true
		if game.state == game.State.GAMEOVER:
			restarting = true
	elif OS.get_environment("GV20_RELEASE") == "1" and game.state == game.State.PLAYING:
		if game.gv.frame == 12:
			key_event(move_key, false)
		elif game.gv.frame == 16:
			key_event(move_key, true)

func _exit_tree() -> void:
	key_event(move_key, false)
	key_event(start_key, false)
