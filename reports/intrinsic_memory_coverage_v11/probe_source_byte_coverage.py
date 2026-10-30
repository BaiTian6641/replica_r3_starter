"""Read-only fitting/public-metadata diagnostic; never parse development labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/intrinsic_memory_coverage_v11'

def pin(path):return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def bound(pin):
    path=ROOT/pin['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256']
    return json.loads(path.read_text())

def main():
    protocol=json.loads((ROOT/'configs/spark_intrinsic_coverage_v11.json').read_text())
    output=ROOT/protocol['arms']['two']['output_root']
    qualification=json.loads((output/'teacher_qualification.json').read_text())
    assert qualification['passed'] and len(qualification['rows'])==32
    fitting=bound(protocol['data']['fit'])
    public=bound(protocol['data_manifest'])
    counts=Counter(byte for row in fitting['rows'] for byte in row['text'].encode('utf-8'))
    entities=public['splits']['development']['public_identity']['entities']
    missing=sorted(set(b for name in entities for b in name.encode('utf-8'))-set(counts))
    result={'format':'replica.intrinsic_memory.byte_coverage_diagnostic.v11',
        'scope':'posthoc_already_opened_fitting_text_and_public_split_metadata_only',
        'protocol':pin(ROOT/'configs/spark_intrinsic_coverage_v11.json'),
        'fitting_data':protocol['data']['fit'],'public_split_manifest':protocol['data_manifest'],
        'teacher_qualification':pin(output/'teacher_qualification.json'),
        'fitting_spans':len(fitting['rows']),'distinct_fitting_input_bytes':len(counts),
        'public_development_entity_bytes_absent_from_fitting_spans':missing,
        'ASCII_display':[chr(b) for b in missing],
        'public_entity_identity_cases':[{'entity':name,'source_bytes_absent_from_fitting':sorted(set(name.encode('utf-8'))-set(counts))} for name in entities],
        'development_labels_parsed':False,'native_model_loads':0,'parameter_updates':0,
        'checkpoint_selection':False,'protocol_changed':False,
        'limits':['Byte absence means no direct source-token exposure during fitting; it does not prove unchanged weights under optimizer decay.',
                  'This coverage mismatch may matter for a byte encoder initialized without pretrained language features. It does not establish the cause of any failure.',
                  'The fixed v11 experiment continues unchanged. Any new producer or corpus requires a separate protocol and binding.']}
    destination=REPORT/'source_byte_coverage.json';assert not destination.exists()
    destination.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('fitting_spans','distinct_fitting_input_bytes','ASCII_display','development_labels_parsed','native_model_loads','parameter_updates')},indent=2))

if __name__=='__main__':main()
