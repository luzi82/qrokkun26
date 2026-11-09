extends Node
## Resolve current poses after movement, before recorders. Area signals lag a tick.
var game

func _ready() -> void:
	process_physics_priority = 50

func _physics_process(_delta: float) -> void:
	if game.state != game.State.PLAYING:
		return
	for bullet in game.bullets.get_children():
		if bullet.is_queued_for_deletion():
			continue
		var radius: float = bullet.get_node("CollisionShape2D").shape.radius
		var player_radius: float = game.player.get_node("CollisionShape2D").shape.radius
		if bullet.position.distance_squared_to(game.player.position) <= pow(radius + player_radius, 2):
			game.player._on_area_entered(bullet)
			return
