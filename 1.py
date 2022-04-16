from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(r'E:\\python_Code\\Classic-ML-C-11\\log-连续\\events.out.tfevents.1650069227.DESKTOP-45M61JV')
ea.Reload()
print(ea.scalars.Keys())

def save_log(log_file, file_path):
    with open(file_path, "w") as f:
        f.write(str(log_file))

reward = ea.scalars.Items('Ant-v2/Reward/Random')
reward_log = []
for (t, s, v) in reward:
    reward_log.append(v)
print(reward_log)

save_log(reward_log, "log2.json")