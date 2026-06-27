"""Esquemas de color ("skins") para el robot DUM, aplicables en runtime.

MuJoCo no necesita texturas de imagen para esto: cada geom tiene un campo
`model.geom_rgba` que el renderer usa como color difuso (con sombreado por la
luz de la escena). Cambiando ese array en vivo se re-pinta el robot sin recargar
el modelo.

Las STL exportadas de Fusion no tienen coordenadas UV, asi que no se pueden
mapear texturas de imagen pieza por pieza de forma prolija. Por eso usamos
color plano por categoria de pieza, que igual queda muy bien con el sombreado.

Uso:
    from skins import apply_skin, SKINS
    apply_skin(model, "realista")   # o "imperio"
"""
from __future__ import annotations

import mujoco


# Clasificacion de cada geom del robot en una "categoria" de pieza.
# (target, ball y piso NO estan aca -> no se re-pintan.)
GEOM_CATEGORY = {
    # torso / cuerpo principal
    "Base_link_geom": "torso",
    "BaseHip_link_geom": "torso",
    "FullBody_link_geom": "torso",
    # hombros
    "BodyShoulderLeft_Link_geom": "shoulder",
    "RightBodyShoulder_Link_geom": "shoulder",
    "LeftShoulderArm_Link_geom": "shoulder",
    "RightShoulderArm_Link_geom": "shoulder",
    # antebrazos
    "LeftForearm_link_geom": "arm",
    "RightForearm_link_geom": "arm",
    # muñecas / dorso de mano
    "WristLeft_link_geom": "hand",
    "RightWrist_link_geom": "hand",
    "LeftFingersLever_link_geom": "hand",
    "RightFingersLever_link_geom": "hand",
    # dedos
    "LeftTopFinger_link_geom": "finger",
    "LeftBotFinger_link_geom": "finger",
    "RigthTopFinger_link_geom": "finger",
    "RightBotFinger_link_geom": "finger",
    # cuello / cabeza / lente
    "Neck_link_geom": "neck",
    "HeadBase_link_geom": "head",
    "LenteExt_link_geom": "lens",
    "LenteInt_link_geom": "lens",
}


# Esquemas de color. rgba en [0,1].
SKINS = {
    # --- REALISTA: Pit Droid (marron/oxido). Torso oscuro para contrastar con el
    #     piso beige; cabeza marron (pedido del usuario). ---
    "realista": {
        "torso":    (0.50, 0.36, 0.22, 1.0),   # marron calido oscuro (contrasta con el piso)
        "shoulder": (0.42, 0.31, 0.21, 1.0),   # marron medio
        "arm":      (0.38, 0.28, 0.19, 1.0),   # marron oscuro
        "hand":     (0.30, 0.23, 0.17, 1.0),   # gunmetal / oxido
        "finger":   (0.26, 0.20, 0.15, 1.0),   # casi negro tierra
        "neck":     (0.40, 0.30, 0.20, 1.0),   # marron
        "head":     (0.50, 0.36, 0.22, 1.0),   # marron = torso
        "lens":     (0.50, 0.36, 0.22, 1.0),   # marron = torso (el plato grande, antes ambar)
    },
    # --- IMPERIO: TODO blanco, solo las muñecas en rojo (pedido del usuario) ---
    "imperio": {
        "torso":    (0.93, 0.94, 0.96, 1.0),   # blanco
        "shoulder": (0.93, 0.94, 0.96, 1.0),   # blanco
        "arm":      (0.93, 0.94, 0.96, 1.0),   # blanco
        "hand":     (0.90, 0.12, 0.12, 1.0),   # ROJO (muñecas)
        "finger":   (0.93, 0.94, 0.96, 1.0),   # blanco
        "neck":     (0.93, 0.94, 0.96, 1.0),   # blanco
        "head":     (0.93, 0.94, 0.96, 1.0),   # blanco
        "lens":     (0.93, 0.94, 0.96, 1.0),   # blanco
    },
}

DEFAULT_SKIN = "realista"


def apply_skin(model, name: str) -> bool:
    """Pinta el robot con el esquema `name`. Devuelve True si se aplico.
    Modifica model.geom_rgba en vivo — surte efecto en el proximo render."""
    scheme = SKINS.get(name)
    if scheme is None:
        return False
    for geom_name, category in GEOM_CATEGORY.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            continue
        rgba = scheme.get(category)
        if rgba is None:
            continue
        model.geom_rgba[gid, :] = rgba
    return True
