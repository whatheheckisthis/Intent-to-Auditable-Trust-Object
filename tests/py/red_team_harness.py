# red_team_harness.py
# Defensive red-team simulator: generates synthetic adversary patterns (poisoning, EDS flood, collusion scans)
# This is a *simulator only* — it creates synthetic packet streams to test detection pipelines.
# DO NOT use for offensive actions.
import csv, random, datetime, json
def generate_synthetic_attack_stream(out_csv, n=1000, seed=1):
    random.seed(seed)
    with open(out_csv,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['timestamp','packet_id','validator_id','event_type','entropy','prov_hash','note'])
        for i in range(n):
            ts = (datetime.datetime.utcnow() - datetime.timedelta(seconds=n-i)).isoformat()+'Z'
            packet_id = f'pkt-{i:05d}'
            validator_id = f'v{random.randint(1,15)}'
            # event types: normal, poisoning_hint, eds_flood_hint, collusion_hint
            r=random.random()
            if r<0.015:
                et='poisoning_hint'
                note='synthetic poisoning pattern'
            elif r<0.03:
                et='eds_flood_hint'
                note='synthetic EDS flood pattern'
            elif r<0.045:
                et='collusion_hint'
                note='synthetic collusion signature pattern'
            else:
                et='normal'
                note='normal packet'
            entropy = round(random.random(),6)
            prov_hash = 'sha256:' + ''.join(random.choice('0123456789abcdef') for _ in range(16))
            w.writerow([ts,packet_id,validator_id,et,entropy,prov_hash,note])

if __name__ == '__main__':
    generate_synthetic_attack_stream('packet_log.csv', n=2000)
