#!/usr/bin/env python3
"""
話者ごとにセリフを分割して別ファイルに保存するスクリプト

Usage:
    python split_speakers.py input.txt [--output-dir DIR]
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict


def parse_dialogue(file_path):
    """対話ファイルをパースして話者ごとにセリフを分類"""
    speakers = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # **A**: セリフ のパターンにマッチ（全角・半角コロン両対応）
    pattern = r'\*\*([A-Z]+)\*\*[：:]\s*(.+?)(?=\n\n|\*\*[A-Z]+\*\*[：:]|$)'

    for match in re.finditer(pattern, content, re.DOTALL):
        speaker = match.group(1)
        dialogue = match.group(2).strip()

        # 空行や余分な改行を削除
        dialogue = dialogue.replace('\n', ' ').strip()

        if dialogue:  # 空でない場合のみ追加
            speakers[speaker].append(dialogue)

    return speakers


def remove_parentheses(text):
    """括弧とその内容、三点リーダー、ダッシュを削除"""
    # 全角括弧を削除
    text = re.sub(r'（[^）]*）', '', text)
    # 半角括弧を削除
    text = re.sub(r'\([^)]*\)', '', text)
    # 【】を削除
    text = re.sub(r'【[^】]*】', '', text)
    # 三点リーダー（……）を削除
    text = re.sub(r'…+', '', text)
    # ダッシュ（——）を削除
    text = re.sub(r'—+', '', text)
    # 複数の空白を1つにまとめる
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def save_speaker_files(speakers, output_dir, base_name):
    """話者ごとにファイルを保存"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for speaker, dialogues in sorted(speakers.items()):
        output_file = output_path / f"{base_name}_speaker_{speaker}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            for dialogue in dialogues:
                # 括弧とその内容を削除
                cleaned_dialogue = remove_parentheses(dialogue)
                if cleaned_dialogue:  # 空でない場合のみ書き込み
                    f.write(cleaned_dialogue + '\n')

        print(f"話者 {speaker}: {len(dialogues)} 個のセリフを {output_file} に保存しました")


def main():
    parser = argparse.ArgumentParser(
        description='話者ごとにセリフを分割して別ファイルに保存'
    )
    parser.add_argument(
        'input_file',
        help='入力テキストファイル'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='出力ディレクトリ (デフォルト: カレントディレクトリ)'
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {args.input_file}")
        return 1

    # 話者ごとにパース
    speakers = parse_dialogue(input_path)

    if not speakers:
        print("警告: 話者が見つかりませんでした")
        return 1

    # ベース名を取得（拡張子なし）
    base_name = input_path.stem

    # ファイル保存
    save_speaker_files(speakers, args.output_dir, base_name)

    print(f"\n合計 {len(speakers)} 人の話者を処理しました")
    return 0


if __name__ == '__main__':
    exit(main())
