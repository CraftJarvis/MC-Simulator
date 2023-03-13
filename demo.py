import minedojo
import time

env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(360,640)
)
obs = env.reset()
done = False
while not done:
    s = time.time()
    act = env.action_space.no_op()
    act[0] = 1    # forward/backward
    obs, reward, done, info = env.step(act)
    e =time.time()
    print("frame rate:",1/(e-s))
env.close()
