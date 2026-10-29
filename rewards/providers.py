"""Image editors through standard APIs or an isolated local Diffusers worker."""
import base64
import io
import os
from pathlib import Path
import requests
from PIL import Image


def build_edit_body(prompt, image, config):
    return {'prompt': prompt, 'image': image}


def image_bytes(reference, timeout):
    if reference.startswith('data:image/'):
        return base64.b64decode(reference.split(',', 1)[1])
    if reference.startswith(('https://', 'http://')):
        response = requests.get(reference, timeout=timeout)
        response.raise_for_status()
        return response.content
    return Path(reference).expanduser().read_bytes()


def request_image_edit(body, config):
    cfg = dict(config['gpt_image2_inference'])
    cfg['_project_dir'] = config.get('_project_dir', str(Path(__file__).resolve().parents[1]))
    backend = cfg.get('backend', 'api')
    if backend in {'external_worker', 'worker', 'subprocess_worker'}:
        from utils.external_image_edit import request_external_image_edit
        return request_external_image_edit(body['prompt'], body['image'], cfg)
    if backend in {'local', 'local_diffusers', 'diffusers'}:
        from utils.local_image_edit import request_local_image_edit
        return request_local_image_edit(body['prompt'], body['image'], cfg)
    if cfg.get('provider', 'openai') != 'openai':
        raise ValueError('API image editing requires provider=openai or a local Diffusers backend')
    from openai import OpenAI
    key_env = cfg.get('api_key_env', 'IMAGE_API_KEY')
    key = os.getenv(key_env)
    if not key:
        raise ValueError(f'Set {key_env} for the image-editing API')
    timeout = float(cfg.get('request_timeout', 300))
    client = OpenAI(api_key=key, base_url=os.getenv(cfg.get('base_url_env', 'IMAGE_BASE_URL')) or None,
                    timeout=timeout)
    # Normalize the source to a supported PNG, preserving full resolution.
    with Image.open(io.BytesIO(image_bytes(body['image'], timeout))) as source:
        image = io.BytesIO()
        source.convert('RGB').save(image, format='PNG')
    image.seek(0); image.name = 'source.png'
    options = {k: cfg[k] for k in ('size', 'quality', 'output_format', 'n') if k in cfg}
    response = client.images.edit(model=cfg.get('model', 'gpt-image-2'), image=image,
                                  prompt=body['prompt'], **options)
    return response.model_dump()
