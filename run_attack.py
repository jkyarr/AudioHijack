import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import hydra
import logging
import omegaconf
import json
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import util
from attack import Attack
from model import LALMFactory
from model.tools import all_tools, tool_prompt


@hydra.main(config_path='config', config_name='run_attack')
def main(cfg: omegaconf.DictConfig) -> None:
    util.set_seed(cfg.seed)
    job_name = HydraConfig.get().job.name
    log = logging.getLogger(job_name)
    log.info('\n' + omegaconf.OmegaConf.to_yaml(cfg))
    device, dtype = util.set_device_dtype(cfg.gpu, cfg.device, cfg.bf16)
    info_fmt = '+ '*10 + '[ {} ]' + ' +'*10
    
    log.info(info_fmt.format('Loading lalm weight'))
    lalm = LALMFactory.create(cfg.lalm.name, cfg.lalm)
    lalm.load(device, dtype)
    
    log.info(info_fmt.format('Loading attack setting'))
    rir_path = os.path.join(cfg.data_dir, 'rir', f'RVB2014_type1_rir_{cfg.attack.rir_type}.wav')
    attacker = Attack(cfg.attack, lalm, rir_path, log)
    audio_type = attacker.carrier.split('_')[0]
    assert audio_type in lalm.audio_type, f'lalm does not support {audio_type} input'
    carrier_dir = os.path.join(cfg.data_dir, 'carrier')
    trial = attacker.carrier + f'_{attacker.carrier_length:.1f}s.wav'
    carrier_audio_path = os.path.join(carrier_dir, trial)
    target_resps = pd.read_json(cfg.target_path, lines=True)
    if not lalm.tool_use:
        target_resps = target_resps[~target_resps['behavior'].str.startswith('Tool Misuse')]
    target_resps = target_resps.values.tolist()
    
    log.info(info_fmt.format('Loading audio-text data'))
    benign_dir = os.path.join(cfg.data_dir, 'benign')
    dataset = pd.read_json(os.path.join(cfg.data_dir, 'benign.jsonl'), lines=True)
    dataset = dataset[dataset['audio_type']==audio_type]
    dataset['speech_prompt'] = dataset['speech_prompt'].apply(lambda x: os.path.join(benign_dir, x))
    dataset.loc[dataset.index[::2], 'prompt_type'] = 'speech_prompt'
    dataset.loc[dataset.index[1::2], 'prompt_type'] = 'text_prompt'
    train_set = dataset.head(100).reset_index(drop=True)
    test_set = dataset.tail(100).reset_index(drop=True)
    if lalm.voice_chat:
        train_prompts = train_set[['speech_prompt', 'text_prompt']].values.tolist()
        test_prompts = test_set.apply(lambda row: row[row['prompt_type']], axis=1).values.tolist()
    else:
        train_prompts = train_set[['text_prompt', 'text_prompt']].values.tolist()
        test_prompts = test_set['text_prompt'].values.tolist()
        test_set['prompt_type'] = 'text_prompt'
    test_set = test_set.to_dict(orient='records')
    
    log.info(info_fmt.format('Generating adversarial examples'))
    result_path = os.path.join(job_name, job_name + '.jsonl')
    wav_dir = os.path.join(job_name, 'wav')
    util.set_dir(wav_dir, False)
    for behavior, label in target_resps:
        if behavior.startswith('Tool Misuse'):
            lalm.tools, lalm.system_tool_prompt = all_tools, tool_prompt
            attacker.accum_grad = True
        else:
            lalm.tools, lalm.system_tool_prompt = None, ''
            attacker.accum_grad = lalm.accum_grad
        adv_audio_path = os.path.join(wav_dir, 'adv_' + behavior + '.wav')
        if os.path.exists(adv_audio_path):
            continue
        else:
            audio_data = util.load_audio(carrier_audio_path, sr=attacker.sample_rate).to(device)
            adv_audio = attacker(audio_data, train_prompts[:attacker.train_size], [], label, wav_dir)
            if cfg.save_audio:
                util.save_audio(adv_audio, adv_audio_path, attacker.sample_rate)
        adv_resp_list, success_list, call_success_list = attacker.evaluate(lalm, adv_audio, test_prompts, [], behavior, label, attacker.batch_size)
        success_rate = sum(success_list) / len(success_list)
        call_success_rates = util.calc_tool_call_metrics(call_success_list)
        log.info(f"Trial={trial} | Behavior={behavior} | Success={success_rate:6.4f} | Call Success=" + ", ".join([f'{sr:6.4f}' for sr in call_success_rates]))
        with open(result_path, 'a', encoding='utf-8') as f:
            for k, data in enumerate(test_set):
                data.update({
                    'trial': behavior + '.wav', 'audio_length': attacker.carrier_length,
                    'behavior': behavior, 'label': label,
                    'adv_resp': adv_resp_list[k], 'success': success_list[k],
                    'invocation_success': call_success_list[k][0]if len(call_success_list) > 0 else None,
                    'syntax_success': call_success_list[k][1]if len(call_success_list) > 0 else None,
                    'execution_success': call_success_list[k][2]if len(call_success_list) > 0 else None,
                    'PISR': success_list[k],
                    'BMSR': call_success_list[k][2]if len(call_success_list) > 0 else None
                })
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    log.info(info_fmt.format('Recording attack result'))
    result = pd.read_json(result_path, lines=True)
    pisr = result.groupby('behavior')['PISR'].mean()
    log.info(pisr)
    # print BMSR
    # - None for non-tool-misuse mishehaviors (need to be evaluated by LLM judges)
    # - Execution success rate for tool-misuse mishehaviors
    bmsr = result.groupby('behavior')['BMSR'].mean()
    log.info(bmsr)


if __name__ == '__main__':
    main()
