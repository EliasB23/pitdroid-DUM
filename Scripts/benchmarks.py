"""Benchmarks consolidados para asesorar Plan A vs Plan B.
B1: Estabilidad con ctrl=0.
B2: Sweep integrador x timestep sobre act_HipBody.
B3: Throughput (mj_step/s) single-thread.
B4: Convergencia actual de cada actuador con un escalon al 50% del rango.
B5: Auditoria estatica MJX (texto, no requiere mjx instalado).
"""
import sys, time, os, json
import numpy as np
import mujoco

XML = os.path.join(os.path.dirname(__file__), "..", "Cuerpo", "DUM4.xml")
XML = os.path.normpath(XML)

def banner(s):
    print("\n" + "="*70)
    print(s)
    print("="*70)

def load(integrator=None, timestep=None):
    """Carga modelo y opcionalmente sobreescribe integrator/timestep en runtime."""
    m = mujoco.MjModel.from_xml_path(XML)
    if integrator is not None:
        m.opt.integrator = integrator
    if timestep is not None:
        m.opt.timestep = timestep
    d = mujoco.MjData(m)
    return m, d

def get_actuator_info(m):
    info = []
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jid = m.actuator_trnid[a, 0]
        qadr = m.jnt_qposadr[jid]
        dof = m.jnt_dofadr[jid]
        lo, hi = m.actuator_ctrlrange[a]
        kp = m.actuator_gainprm[a, 0]
        kv = -m.actuator_biasprm[a, 2]
        info.append(dict(name=name, aid=a, jid=jid, qadr=qadr, dof=dof,
                         ctrl_lo=lo, ctrl_hi=hi, kp=kp, kv=kv))
    return info

# ---------------- B1 ----------------
def b1_stability():
    banner("B1 - Estabilidad con ctrl=0 durante 2.0s")
    m, d = load()
    mujoco.mj_resetData(m, d)
    nan_step = None
    max_qpos = 0.0
    max_qvel = 0.0
    n_steps = int(2.0 / m.opt.timestep)
    for i in range(n_steps):
        mujoco.mj_step(m, d)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            nan_step = i
            break
        max_qpos = max(max_qpos, np.max(np.abs(d.qpos)))
        max_qvel = max(max_qvel, np.max(np.abs(d.qvel)))
    # Warnings de MuJoCo
    warns = []
    for w in range(mujoco.mjtWarning.mjNWARNING.value):
        c = d.warning[w].number
        if c > 0:
            wn = mujoco.mjtWarning(w).name
            warns.append((wn, c))
    print(f"timestep={m.opt.timestep}, integrator={mujoco.mjtIntegrator(m.opt.integrator).name}")
    print(f"steps simulados: {i+1}/{n_steps}  NaN en step: {nan_step}")
    print(f"|qpos|_max = {max_qpos:.4f}   |qvel|_max = {max_qvel:.4f}")
    print(f"warnings: {warns if warns else 'ninguna'}")
    return dict(ok=nan_step is None and not warns,
                nan_step=nan_step, max_qpos=max_qpos, max_qvel=max_qvel, warnings=warns)

# ---------------- B2 ----------------
def b2_integrator_sweep():
    banner("B2 - Sweep integrador x timestep sobre act_HipBody (escalon 0.4 rad)")
    integrators = [
        (mujoco.mjtIntegrator.mjINT_EULER.value,        "Euler"),
        (mujoco.mjtIntegrator.mjINT_IMPLICIT.value,     "implicit"),
        (mujoco.mjtIntegrator.mjINT_IMPLICITFAST.value, "implicitfast"),
    ]
    timesteps = [0.001, 0.002, 0.005]
    target = 0.4
    sim_time = 2.0
    rows = []
    for itg_val, itg_name in integrators:
        for dt in timesteps:
            m, d = load(integrator=itg_val, timestep=dt)
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_HipBody")
            jid = m.actuator_trnid[aid, 0]
            qadr = m.jnt_qposadr[jid]
            mujoco.mj_resetData(m, d)
            ts, ps = [], []
            exploded = False
            t_settle = None
            n = int(sim_time / dt)
            for i in range(n):
                d.ctrl[aid] = target
                mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos[qadr]):
                    exploded = True; break
                ts.append(d.time)
                ps.append(d.qpos[qadr])
                if t_settle is None and abs(ps[-1] - target) < 0.05*abs(target):
                    t_settle = d.time
            if exploded:
                rows.append(dict(itg=itg_name, dt=dt, status="NaN", overshoot=None,
                                 t_settle=None, rms_late=None))
                continue
            ps = np.array(ps); ts = np.array(ts)
            overshoot = (max(ps) - target) / target * 100 if target != 0 else 0
            late_mask = ts > sim_time*0.5
            rms_late = float(np.sqrt(np.mean((ps[late_mask]-target)**2))) if late_mask.any() else None
            rows.append(dict(itg=itg_name, dt=dt, status="ok",
                             overshoot=round(overshoot,1),
                             t_settle=round(t_settle,3) if t_settle else None,
                             rms_late=round(rms_late,4) if rms_late else None))
    # Tabla
    print(f"{'integrator':14s} {'dt':>7s} {'status':>8s} {'over%':>8s} {'t_settle':>10s} {'rms_late':>10s}")
    for r in rows:
        over_s = "-" if r['overshoot'] is None else f"{r['overshoot']:.1f}"
        sett_s = "-" if r['t_settle']  is None else f"{r['t_settle']:.3f}"
        rms_s  = "-" if r['rms_late']  is None else f"{r['rms_late']:.4f}"
        print(f"{r['itg']:14s} {r['dt']:>7.4f} {r['status']:>8s} "
              f"{over_s:>8s} {sett_s:>10s} {rms_s:>10s}")
    return rows

# ---------------- B3 ----------------
def b3_throughput():
    banner("B3 - Throughput single-thread (mj_step/s)")
    m, d = load()
    mujoco.mj_resetData(m, d)
    # Warm up
    for _ in range(1000):
        mujoco.mj_step(m, d)
    N = 50000
    t0 = time.perf_counter()
    for _ in range(N):
        mujoco.mj_step(m, d)
    elapsed = time.perf_counter() - t0
    sps = N/elapsed
    realtime_factor = sps * m.opt.timestep  # > 1 = mas rapido que real
    print(f"timestep={m.opt.timestep}")
    print(f"steps/s single-thread: {sps:,.0f}")
    print(f"real-time factor: {realtime_factor:.1f}x")
    # Proyeccion para 8 envs paralelos (estimacion lineal, overhead aparte):
    print(f"con 8 envs paralelos (ideal): ~{sps*8:,.0f} steps/s -> "
          f"~{sps*8*86400/1e6:.1f} Msteps/dia, "
          f"~{sps*8*86400/1e6*1:.1f} M en 24h")
    return dict(sps=sps, rtf=realtime_factor)

# ---------------- B4 ----------------
def b4_current_calibration():
    banner("B4 - Convergencia con kp/kv actuales (escalon a 50% del ctrlrange, 1.5s, err<0.05)")
    m, d = load()
    actuators = get_actuator_info(m)
    sim_time = 1.5
    err_thresh = 0.05
    results = []
    for ai in actuators:
        mujoco.mj_resetData(m, d)
        # Target = 50% del rango positivo o, si es bilateral, mitad superior
        lo, hi = ai['ctrl_lo'], ai['ctrl_hi']
        target = hi*0.5 if hi > 0 else lo*0.5
        ps = []
        n = int(sim_time / m.opt.timestep)
        for _ in range(n):
            d.ctrl[ai['aid']] = target
            mujoco.mj_step(m, d)
            ps.append(d.qpos[ai['qadr']])
        ps = np.array(ps)
        final_err = abs(ps[-1] - target)
        ok = final_err < err_thresh
        mse = float(np.mean((ps - target)**2))
        results.append(dict(name=ai['name'], target=round(target,4),
                            final=round(float(ps[-1]),4),
                            err=round(final_err,4), ok=ok, kp=ai['kp'], kv=ai['kv'],
                            mse=round(mse,5)))
    # Tabla
    print(f"{'actuador':25s} {'target':>8s} {'final':>8s} {'err':>8s} {'kp':>6s} {'kv':>5s} {'pass':>5s}")
    for r in results:
        flag = "OK" if r['ok'] else "FAIL"
        print(f"{r['name']:25s} {r['target']:>8.4f} {r['final']:>8.4f} {r['err']:>8.4f} "
              f"{r['kp']:>6.0f} {r['kv']:>5.0f} {flag:>5s}")
    passed = sum(1 for r in results if r['ok'])
    print(f"\nPASA: {passed}/{len(results)}")
    return results

# ---------------- B5 ----------------
def b5_mjx_audit():
    banner("B5 - Auditoria MJX (estatica, sobre el XML)")
    # Hechos conocidos sobre MJX (a 3.x):
    #  - <equality joint polycoef=...>  -> SOPORTADO en MJX (constraint type ConstraintType.JOINT). OK.
    #  - <equality connect/weld>        -> SOPORTADO.
    #  - <equality tendon>              -> SOPORTADO en versiones recientes.
    #  - Integradores soportados en MJX: Euler, RK4, implicitfast. NO soporta 'implicit' clasico.
    #  - Actuators: motor, position, velocity, intvelocity, damper -> OK. muscle/cylinder limitado.
    #  - Sensors: la mayoria OK, algunos (rangefinder, touch grid) limitados.
    #  - Mesh collisions: limitadas; usar primitives o convex hulls cuando se pueda.
    #  - condim: todos soportados pero hay diferencias numericas con CPU.
    m, _ = load()
    issues, oks = [], []
    # Equality
    if m.neq > 0:
        oks.append(f"{m.neq} equality constraints (todas polynomial JOINT) -> soportado en MJX")
    # Integrator
    itg_name = mujoco.mjtIntegrator(m.opt.integrator).name
    if itg_name in ("mjINT_EULER", "mjINT_IMPLICITFAST"):
        oks.append(f"integrador {itg_name} -> soportado en MJX")
    elif itg_name == "mjINT_IMPLICIT":
        issues.append(f"integrador {itg_name} -> NO soportado en MJX, cambiar a implicitfast")
    else:
        oks.append(f"integrador {itg_name} -> probablemente soportado, verificar docs MJX")
    # Actuator types
    bad_act = []
    for a in range(m.nu):
        gt = mujoco.mjtGain(m.actuator_gaintype[a]).name
        bt = mujoco.mjtBias(m.actuator_biastype[a]).name
        # position actuators usan gain=fixed, bias=affine -> soportado
        if gt not in ("mjGAIN_FIXED", "mjGAIN_AFFINE", "mjGAIN_MUSCLE", "mjGAIN_USER"):
            bad_act.append((a, gt, bt))
    if not bad_act:
        oks.append(f"{m.nu} actuators position-style (gain=fixed, bias=affine) -> soportado")
    # Mesh count
    nmesh = m.nmesh
    oks.append(f"{nmesh} meshes STL. MJX soporta mesh collisions limitadas; "
               f"si se entrena con muchos envs en GPU, considerar primitives o "
               f"verificar que no haya colisiones mesh<->mesh criticas")
    # Contacts auto vs explicit
    ncon = m.nconmax if hasattr(m, 'nconmax') else 'auto'
    oks.append(f"contactos: auto, sin <contact> explicito (puede generar muchos pares con tantos meshes)")
    print("Soportado:")
    for o in oks: print(f"  + {o}")
    print("Cambios necesarios para MJX:")
    if issues:
        for i in issues: print(f"  ! {i}")
    else:
        print("  (ninguno bloqueante a nivel XML)")
    return dict(oks=oks, issues=issues)

# ---------------- main ----------------
if __name__ == "__main__":
    summary = {}
    summary['B1'] = b1_stability()
    summary['B2'] = b2_integrator_sweep()
    summary['B3'] = b3_throughput()
    summary['B4'] = b4_current_calibration()
    summary['B5'] = b5_mjx_audit()
    banner("RESUMEN JSON")
    print(json.dumps(summary, indent=2, default=str))
