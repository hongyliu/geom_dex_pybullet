from gym.envs.registration import register


for reward_type in ["dense", "sparse"]:
    suffix = "Dense" if reward_type == "dense" else ""

    # Difficult mode not needed for Block
    kwargs = {"reward_type": reward_type}

    register(
        id="ShadowHandBlock{}-v1".format(suffix),
        entry_point="shadowhand_gym.envs:ShadowHandBlockEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )
