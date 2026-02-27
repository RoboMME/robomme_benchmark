import gymnasium as gym

from ..logging_utils import logger

class FailAwareWrapper(gym.Wrapper):
    """
    统一在最外侧接住所有的异常崩溃 (例如 IK Fail)。将抛出的代码 Error 转化为状态码 info = {"status": "error"} 并终止 episode，保证所有外围执行脚本无需手写 try except。
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            self._last_obs = obs
            return obs, reward, terminated, truncated, info
        except Exception as e:
            # 记录异常方便溯源调试
            logger.error(f"环境中途阻断执行发生异常: {str(e)}")
            
            # 直接触发 terminated 的退出机制，且向 info 注入错误标志即可
            return (
                None,            #no obs
                0.0,            # 惩罚性 reward 或维持 0
                True,           # terminated: True，令循环切断
                False,          # truncated
                {
                    "status": "error", 
                    "error_message": f"FailAwareWrapper caught specific exception: {str(e)}",
                    "exception_type": type(e).__name__
                }
            )
