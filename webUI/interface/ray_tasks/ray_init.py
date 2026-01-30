
def init_ray():
    try:
        import ray
    except ImportError:
        print("Ray is not installed, skipping initialization.")
        return None

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    return ray

