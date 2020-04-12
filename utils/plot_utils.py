import numpy as np
import matplotlib.pyplot as plt

def model_rollout_acc(T, env, num_steps=50):
    """L2 loss of rollout for model T over time horizon

    Arguments: 

        T1 (function) - transition model 1 (s, a) -> s'
        env (gym.Env) - OpenAI gym env
        num_steps (int) - number of steps to compare over
    """

    loss = np.zeros((100, num_steps))

    # 100 Trials
    for i in range(100):
        state = env.reset()
        
        for s in range(num_steps):
            action = env.action_space.sample()

            next_state, _, done, _ = env.step(action)
            pred_next_state = T((state, action))

            diff = next_state - pred_next_state
            loss[i, s] = diff.dot(diff)
    
    plt.plot(np.mean(loss, axis=0))
    plt.title(f"Average loss for rollout of length {num_steps}")
    plt.show()
    
def generate_video(imgs, filename):
    from PIL import Image
    from matplotlib import cm

    pil_imgs = []
    for img in imgs:
        pil_imgs.append(Image.fromarray(img))
    
    pil_imgs[0].save(filename,
               save_all=True,
               append_images=pil_imgs[1:],
               duration=100,
               loop=5)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


