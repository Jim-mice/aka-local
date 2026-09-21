import hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]))
from lab.core.delivery import validate_source_delivery
root=pathlib.Path(__file__).resolve().parents[3]; target=root/'targets/megatron_5be9626/moe_native_sequential_expert_compute'; abi=json.loads((target/'required_abi.json').read_text()); opt=target/'optimization_contract.json'; marker=abi['marker']['literal']; proto=abi['prototype']; symbol=abi['required_symbol']; expected=hashlib.sha256(opt.read_bytes()).hexdigest(); results=[]
for name in sys.argv[1:] or ['M1','M2']:
 ep=root/'campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute'/f'episode_{name}'; src=ep/'candidate.cu'; text=src.read_text(); ok,cls,detail=validate_source_delivery(text,marker_literal=marker,prototype=proto,required_symbol=symbol); results.append({'episode':name,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'optimization_contract_sha256':expected,'ok':ok,'class':cls,'detail':detail})
print(json.dumps({'results':results},indent=2))
