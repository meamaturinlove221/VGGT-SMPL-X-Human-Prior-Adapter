"""从实际教师和学生端点登记可打开的 PLY。"""
import csv
from pathlib import Path

from replay import FRAMES, NEW_FRAMES, OFFICIAL, OUT, PREVIOUS, STUDENT, load, sample, student_farm


def main():
    files, summary = [], []

    def register(sid, run, evaluated, role, path, receipt, checkpoint=''):
        path = Path(path)
        if not path.is_file() or not Path(receipt).is_file():
            raise FileNotFoundError(path if not path.is_file() else receipt)
        files.append(dict(frame=sid,run=run,evaluated_frame=evaluated,artifact=role,ply=str(path),
                          bytes=path.stat().st_size,receipt=str(receipt),checkpoint=str(checkpoint)))
        return path

    def outputs(sid, run, receipt):
        actual = load(receipt)
        folder = Path(actual['output'])
        names = {'RAW':'source_RAW_all_foreground.ply'}
        if actual['surface_status'] == 'EXPORTED_ALL_COMPONENTS':
            names.update(Base='base_mesh.ply', Final='clothed_mesh.ply', Gaussians='clothed_gaussians.ply')
        return {key:register(sid, run, actual['id'], key, folder/name, receipt, actual['checkpoint'])
                for key,name in names.items()}

    with (PREVIOUS/'SIX_DN_GW_2M_DISPLAY_INDEX.csv').open(encoding='utf-8-sig',newline='') as stream:
        archive = {r['sample']:r for r in csv.DictReader(stream)}
    for code in FRAMES:
        sid = sample(code)
        teacher = None
        current = None
        parent = None
        if code == '0019':
            teacher = OFFICIAL
            check = STUDENT.parent/'display_dense/RAW_BODY_DENSE_RGB/RECEIPT.json'
            display = load(check)
            previous_raw = Path(display['source_RAW_PLY'])
            previous_2m = display['output']
        else:
            display = archive[sid]
            previous_raw = Path(display['source_raw'])
            check = Path(display['receipt'])
            previous_2m = display['ply']
            if code in ('0008','0047','0102'):
                teacher = (OUT if code=='0102' else PREVIOUS)/f'GW40_{code}'
            elif code in NEW_FRAMES:
                candidate = OUT/f'GW40_{code}'
                if ((candidate/'GW40_ACTUAL_ENDPOINT.json').is_file()
                        and (candidate/'human_body_only/BODY_ONLY_RECEIPT.json').is_file()):
                    teacher = candidate
        historic = outputs(sid, 'ORIGINAL_0019_DN_GW_U200', previous_raw.with_name('INFERENCE.json'))
        if historic['RAW'] != previous_raw:
            raise ValueError(f'{sid} 历史显示的 RAW 归属不同')
        historic_2m = register(sid, 'ORIGINAL_0019_DN_GW_U200', sid, '2M_DISPLAY', previous_2m, check,
                               STUDENT/'DN_GW/checkpoint_u0200.pt')
        if teacher:
            model = teacher/'GW_OFFICIAL_DIRECT_DEFAULT'
            body = teacher/'human_body_only'
            receipt = teacher/('RUN_CONTRACT.json' if code=='0019' else 'GW40_ACTUAL_ENDPOINT.json')
            checkpoint = model/'chkpnt30000.pth'
            for path in sorted(model.rglob('*.ply')):
                register(sid, 'GW40_TEACHER', sid, path.relative_to(model).as_posix(), path, receipt, checkpoint)
            for path in sorted(body.glob('*.ply')):
                register(sid, 'GW40_TEACHER', sid, path.name, path, body/'BODY_ONLY_RECEIPT.json', checkpoint)
            parent = body/'GW_HUMAN_BODY_ONLY_DISPLAY.ply'
        farm = student_farm(code)
        evaluated = farm/'evaluation/DN_GW_U200/EVALUATION_RECEIPT.json'
        if code in ('0008','0047','0102') or evaluated.is_file():
            evals = load(evaluated)
            learner = 'ISOLATED_DIAGNOSTIC_DN_GW_U200' if code in NEW_FRAMES else 'MATCHED_DN_GW_U200'
            for record in evals:
                payload = outputs(sid, learner, Path(record['output'])/'INFERENCE.json')
                if record['id']==sid:
                    current = payload
            check = farm/'display_2M'/sid.replace('/','__')/'RECEIPT.json'
            if code not in NEW_FRAMES or check.is_file():
                display = load(check)
                if display['sample_id'] != sid or Path(display['source_RAW_PLY']) != current['RAW']:
                    raise ValueError(f'{sid} 匹配显示的 RAW 归属不同')
                current['2M'] = register(sid,learner,sid,'2M_DISPLAY',display['output'],check,display['checkpoint'])
        summary.append((sid,historic['RAW'],historic_2m,parent,current))

    index = OUT/'TEACHER_STUDENT_PLY_INDEX.csv'
    with index.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=files[0])
        writer.writeheader()
        writer.writerows(files)

    def link(path, label):
        return f'[{label}]({path.as_posix()})' if path else '—'

    lines = ['# 同帧教师与学生 PLY','',
             '完整索引逐项列出绝对路径、字节数、实际回执与检查点。独立诊断分支的本帧评价属于训练内评价。 [完整 PLY 索引](TEACHER_STUDENT_PLY_INDEX.csv)','',
             '| 帧 | 原 0019 学生 RAW / 2M | 同帧教师与回执 | 同帧匹配学生 RAW / Base / Final / 2M |',
             '|---|---|---|---|']
    for sid, raw, dense, teacher, current in summary:
        assets = ' · '.join(link(current[key],key) for key in ('RAW','Base','Final','2M') if key in current) if current else '—'
        teacher_assets = '—'
        if teacher:
            run = teacher.parents[1]
            model = run/'GW_OFFICIAL_DIRECT_DEFAULT'
            teacher_files = {'完整 RAW':model/'RAW_UNFILTERED.ply',
                             'FULL':model/'mesh_ours_2pivots_post_texture_refined_999.ply',
                             'BODY':teacher, '相机':run/'FRAME_INPUT.json',
                             '筛选':teacher.with_name('BODY_ONLY_RECEIPT.json')}
            teacher_assets = ' · '.join(link(path,label) for label,path in teacher_files.items() if path.is_file())
        lines.append(f"| {sid} | {link(raw,'RAW')} · {link(dense,'2M')} | {teacher_assets} | {assets} |")
    (OUT/'TEACHER_STUDENT_PLY_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(index, len(files), 'PLY entries')


if __name__ == '__main__':
    main()
