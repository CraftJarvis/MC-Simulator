import minedojo


if __name__ == "__main__":
    env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=(288, 512),
        world_seed=123,
        seed=42,
    )

    print(f"[INFO] Create a task with prompt: {env.task_prompt}")

    env.reset()
    for _ in range(20):
        obs, reward, done, info = env.step(env.action_space.no_op())
    env.close()

    print("[INFO] Installation Success")
