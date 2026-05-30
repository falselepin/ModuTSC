import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modutsc.env.sumo_env import SumoEnv

def main():
    env = SumoEnv()
    env.launch({
        "roadnet_file": "data/Monaco/roadnet.net.xml",
        "flow_file": "data/Monaco/flow_0.rou.xml",
        "sim_max_time": 3600,
        "decision_interval": 5,
    })
    ids = env.ids()
    print(f"Found {len(ids)} intersections: {ids}")
    for jid in ids:
        pc = env.phase_count(jid)
        import traci
        logic = traci.trafficlight.getAllProgramLogics(jid)
        if logic:
            phases = logic[0].getPhases()
            phase_states = [p.state for p in phases]
            print(f"  {jid}: phase_count={pc}, total phases={len(logic[0].getPhases())}, states={phase_states}")
        else:
            print(f"  {jid}: phase_count={pc}, NO LOGIC")
    env.close()

if __name__ == "__main__":
    main()
