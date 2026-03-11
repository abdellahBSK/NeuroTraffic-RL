# NeuroTraffic-RL Architecture Diagram

## System Overview

```mermaid
flowchart TD
    subgraph Config["⚙️ Configuration (YAML)"]
        IC[intersection_casablanca.yaml]
        TR[training.yaml]
        SM[simulation.yaml]
    end

    subgraph SUMO["🚗 SUMO Simulator"]
        NET[casablanca_intersection.net.xml]
        ROU[casablanca_intersection.rou.xml]
        SUB[SUMO subprocess]
    end

    subgraph Env["🌍 Environment Layer (env/)"]
        OB[ObservationBuilder\n26-feature vector]
        PM[PhaseManager\nYELLOW injection]
        RC[RewardCalculator\nα·wait + β·tp - γ·queue - δ·switch]
        SE[SumoIntersectionEnv\nGymnasium.Env]
    end

    subgraph Agents["🤖 Agents"]
        PPO[PPOAgent\nStable-Baselines3]
        FC[FixedCycleAgent\n30s/3s baseline]
    end

    subgraph Training["🏋️ Training"]
        TR_SCRIPT[training/train.py]
        EV_SCRIPT[training/evaluate.py]
        CB[MetricsCallback\nBestModelCallback]
    end

    subgraph Dashboard["📊 Dashboard"]
        MS[MetricsStore\nJSON on disk]
        ST[Streamlit App\ndashboard/app.py]
    end

    subgraph Phase2["🔮 Phase 2 Scaffold"]
        BUS[AbstractMessageBus\nNoOpMessageBus]
    end

    Config --> SE
    SUMO -- TraCI --> OB
    OB --> SE
    PM --> SE
    RC --> SE
    SE -- obs,reward --> PPO
    PPO -- action --> PM
    PPO -- train --> TR_SCRIPT
    TR_SCRIPT --> CB
    CB --> MS
    MS --> ST
    SE --> MS
    FC -- compare --> EV_SCRIPT
    PPO -- compare --> EV_SCRIPT
    SE -. Phase 2 .-> BUS
    PPO -. Phase 2 .-> BUS
```

## Phase 1 → Phase 2 Transition

```mermaid
graph LR
    A["Phase 1\nSingle Intersection\nNoOpMessageBus"] 
    -->|"Replace NoOpMessageBus\nwith RedisMessageBus"| 
    B["Phase 2\nMulti-Intersection\nCoordinated Agents"]
    
    B --> C["Phase 3\nSmart City\nHierarchical Control"]
```
