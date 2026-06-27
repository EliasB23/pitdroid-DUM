import multiprocessing as mp
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import time
XML_PATH = "Cuerpo\DUM4.xml"  

def search_one_actuator(act_name, kp_values, kd_values, damping_values, gears,
                        target_dict, sim_time, error_threshold):
    """Worker independiente: carga su propia copia del modelo."""
    # Carga silenciosa del modelo
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    act_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    joint_id = model.actuator_trnid[act_id, 0]
    dof_id   = model.jnt_dofadr[joint_id]
    qpos_adr = model.jnt_qposadr[joint_id]
    
    # El valor objetivo específico para este actuador
    target = target_dict[act_name]

    best_mse = float('inf')
    winner   = None
    EQUALITY_FINGERS = {
    "act_LeftLever_Slider":  ["LeftFingerTop_joint", "LeftFingerBot_joint"],
    "act_RightLever_Slider": ["RightFingerTop_joint", "RightFingerBot_joint"],
}

    # Resolvé los qpos_adr de los fingers vinculados (solo si aplica)
    finger_ids = {}
    if act_name in EQUALITY_FINGERS:
        for jname in EQUALITY_FINGERS[act_name]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            finger_ids[jname] = model.jnt_qposadr[jid]
    # Flush=True asegura que el print salga del proceso hijo a la terminal principal
    print(f"[START] {act_name}", flush=True)

    for g in gears:
        for kp in kp_values:
            print(f'{act_name} prueba {kp}', flush=True)
            for kd in kd_values:
                for d in damping_values:                    
                    model.actuator_gear[act_id, 0]    = g
                    model.actuator_gainprm[act_id, 0] = kp
                    model.actuator_biasprm[act_id, 1] = -kp
                    model.actuator_biasprm[act_id, 2] = -kd
                    model.dof_damping[dof_id]         = d

                    mujoco.mj_resetData(model, data)
                    temp_qpos = []
                    while data.time < sim_time:
                        # if data.time < 1.5:
                            data.ctrl[act_id] = target
                            mujoco.mj_step(model, data)
                            temp_qpos.append(data.qpos[qpos_adr])
                            final_pos = temp_qpos[-1]
                            mse = np.mean((np.array(temp_qpos) - target)**2)
                            
                            if abs(final_pos - target) <= error_threshold:
                                if mse < best_mse:
                                    best_mse = mse
                                    winner = {'kp': kp, 'kd': kd, 'd': d, 'g': g, 'final_val': final_pos}
                        # else:
                        #     data.ctrl[act_id] = 0.0
                        #     mujoco.mj_step(model, data)
                        #     temp_qpos.append(data.qpos[qpos_adr])

                        #     final_pos = temp_qpos[-1]
                        #     mse = np.mean((np.array(temp_qpos) - 0.0)**2)

                        #     if abs(final_pos - 0.0) <= error_threshold:
                        #         if mse < best_mse:
                        #             best_mse = mse
                        #             winner = {'kp': kp, 'kd': kd, 'd': d, 'g': g, 'final_val': final_pos}

    # Telemetría final con el ganador para retornar datos de gráfica
    if winner:
        model.actuator_gear[act_id, 0]    = winner['g']
        model.actuator_gainprm[act_id, 0] = winner['kp']
        model.actuator_biasprm[act_id, 1] = -winner['kp']
        model.actuator_biasprm[act_id, 2] = -winner['kd']
        model.dof_damping[dof_id]         = winner['d']
        TARGET_SWITCH_TIME = 1.5
        mujoco.mj_resetData(model, data)
        t_log, p_log, f_log = [], [], []
        finger_logs = {jname: [] for jname in finger_ids}
        target_original = target_dict[act_name]
        while data.time < sim_time:
            data.ctrl[act_id] = target_original # if data.time < TARGET_SWITCH_TIME else 0.0
            mujoco.mj_step(model, data)
            t_log.append(data.time)
            p_log.append(data.qpos[qpos_adr])
            f_log.append(data.actuator_force[act_id])
            for jname, qadr in finger_ids.items():
                finger_logs[jname].append(data.qpos[qadr])
                        
        print(f"[SUCCESS] {act_name} optimizado.", flush=True)
        target_log = [target_original for t in t_log]    # if t < TARGET_SWITCH_TIME else 0.0
        return {
            'act_name': act_name, 'winner': winner, 'mse': best_mse,
            't': t_log, 'p': p_log, 'f': f_log, 'target': target_log,
            'finger_logs': finger_logs,
            'finger_ranges': {
                jname: (model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname), 0],
                        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname), 1])
                for jname in finger_ids
            }
            }
    else:
        print(f"[FAILED] {act_name} sin solución.", flush=True)
        return {'act_name': act_name, 'winner': None}

if __name__ == "__main__":
    actuators_to_test = ["act_LeftShoulderArm", "act_Neck", "act_HipBody", #"act_BaseHip"
                         "act_HeadBase", "act_HeadRot", "act_RightShoulderArm",
                         "act_LeftForearm", "act_RightForearm", "act_LeftWrist",
                         "act_RightWrist",
                         "act_LeftLever_Slider", "act_RightLever_Slider"]

    # Espacio de búsqueda
    kp_values      = np.arange(2000, 3500, 250)
    kd_values      = np.arange(0, 60, 20)
    damping_values = np.arange(0, 60, 10)
    gears          = [1, 5]

    target_val = {
        # "act_BaseHip": 0.12} 
    "act_HipBody": 0.40, "act_Neck": 0.50, 
    "act_HeadBase": 0.50, "act_HeadRot": 0.50, "act_LeftShoulderArm": 0.07,
    "act_RightShoulderArm": 0.20, "act_LeftForearm": 1.0, "act_RightForearm": 0.5,
    "act_LeftWrist": 1.0, "act_RightWrist": 1.0, 
    "act_LeftLever_Slider": 0.012, "act_RightLever_Slider": 0.012,
    }
    
    

    sim_time = 3.2
    error_threshold = 0.03

    N_WORKERS = max(1, mp.cpu_count() - 1)
    start_time= time.time()
    print(start_time)
    worker_func = partial(
        search_one_actuator,
        kp_values=kp_values, kd_values=kd_values, damping_values=damping_values,
        gears=gears, target_dict=target_val, sim_time=sim_time, error_threshold=error_threshold
    )

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(worker_func, actuators_to_test)

    # Filtrar solo los exitosos para graficar
    valid_results = [r for r in results if r['winner'] is not None]
    valid_results.sort(key=lambda x: x['mse'])
    end = (time.time() - start_time)

    if valid_results:
        display_count = len(valid_results)
        cols = 2
        rows = int(np.ceil(display_count / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axs = axs.flatten() if display_count > 1 else [axs]

        for i, res in enumerate(valid_results):
            has_fingers = bool(res.get('finger_logs'))
            ax = axs[i]
            w = res['winner']
            ax.plot(res['t'], [res['target']]*len(res['t']), 'r--', label='Target')
            ax.plot(res['t'], res['p'], 'b-', label='Posición')
            if has_fingers:
                ax.set_title(f"{res['act_name']} — Gear:{w['g']} | KP:{w['kp']} | KD:{w['kd']} | D:{w['d']}", fontsize=9)
            else:
                ax.set_title(f"{res['act_name']} — Gear:{w['g']} | KP:{w['kp']} | KD:{w['kd']} | D:{w['d']}", fontsize=9)
            
            if has_fingers:
                colors = ['darkorange', 'purple']
                styles = ['-', '--']
                for (jname, flog), col, sty in zip(res['finger_logs'].items(), colors, styles):
                    lo, hi = res['finger_ranges'][jname]
                    ax.plot(res['t'], flog, color=col, ls=sty, lw=1.2, label=f"{jname.split('_')[0]+jname.split('_')[1][-3:]} (rad)")
                    # línea de límite físico del finger
                    limit_val = hi if flog[-1] > 0 else lo
                    ax.axhline(limit_val, color=col, ls=':', lw=0.8, alpha=0.6, label=f"límite {jname.split('_')[0][-1]+jname.split('_')[1][-3:]}: {limit_val:.3f}")
            # Eje secundario para fuerza
            ax2 = ax.twinx()
            ax2.plot(res['t'], res['f'], 'g-', alpha=0.3, label='Fuerza')
            
            ax.set_title(f"{res['act_name']} — Gear:{w['g']} | KP:{w['kp']} | KD:{w['kd']} | D:{w['d']}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Tabla Resumen
        summary = []
        for r in valid_results:
            summary.append({'Actuador': r['act_name'], **r['winner'], 'MSE': r['mse']})
        print("\n" + "="*60)
        print(pd.DataFrame(summary).to_string(index=False))
        print("="*60)
    else:
        print("No se encontraron configuraciones válidas.")
    print(f'Total Time: {end}')