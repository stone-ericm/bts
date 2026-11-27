import sys,collections
exec(open("/Users/eric/bts-runs/fresh15/closing3/normdiff.py").read().split("a,b=")[0])
a=collections.Counter(norm(open(sys.argv[1],errors="replace").read())); b=collections.Counter(norm(open(sys.argv[2],errors="replace").read()))
only_a=a-b; only_b=b-a
print(f"only FINAL2: {sum(only_a.values())}  only FINAL3: {sum(only_b.values())}")
for k,v in list(only_a.items())[:int(sys.argv[3])]: print(" -",v,k[:230])
for k,v in list(only_b.items())[:int(sys.argv[3])]: print(" +",v,k[:230])
