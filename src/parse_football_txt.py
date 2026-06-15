"""
统一解析器 v3 — 支持所有世界杯 openfootball 格式 (1990-2022)

格式汇总:
  1990:    "9 June   Italy    1-0   Austria   @ Venue"
  1994:    日期行"June 18" + 下一行"  USA  1-1  Switzerland  @ Venue"
  1998:    "10 June  Brazil  2-1  Scotland  @ Venue"
  2002:    "31 May   France  0-1  Senegal   @ Venue"
  2006:    "Fri Jun 9  Germany  4-2 (2-1)  Costa Rica  @ Venue"
  2010:    "Fri Jun 11  South Africa  1-1 (0-0)  Mexico  @ Venue"
  2014:    日期行 + "17:00 UTC-3  Brazil v Croatia  3-1 (1-1)  @ Venue"
  2014 KO: 日期行 + "... 1-1 a.e.t. (1-1, 1-1), 3-2 pen.  Chile  @ Venue"
  2018:    日期行 + "17:00 UTC+3  France  4-3 (1-1)  Argentina  @ Venue"
  2022:    日期行 + "19:00  Qatar  0-2 (0-2)  Ecuador  @ Venue"
"""
import re, csv, os
from collections import Counter

MONTH_MAP = {
    'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06',
    'july':'07','august':'08','september':'09','october':'10','november':'11','december':'12',
    'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
    'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12',
}

TEAM_MAP = {
    'england':'England','spain':'Spain','france':'France','germany':'Germany',
    'deutschland':'Germany','west germany':'Germany','portugal':'Portugal',
    'netherlands':'Netherlands','belgium':'Belgium','croatia':'Croatia',
    'italy':'Italy','switzerland':'Switzerland','sweden':'Sweden','denmark':'Denmark',
    'poland':'Poland','serbia':'Serbia','serbia & montenegro':'Serbia',
    'serbia and montenegro':'Serbia','yugoslavia':'Serbia','soviet union':'Russia',
    'russia':'Russia','turkey':'Turkiye','turkiye':'Turkiye','austria':'Austria',
    'scotland':'Scotland','czech republic':'Czechia','czechia':'Czechia',
    'czechoslovakia':'Czechia','romania':'Romania','bulgaria':'Bulgaria',
    'greece':'Greece','norway':'Norway','wales':'Wales','ireland':'Ireland',
    'northern ireland':'Northern Ireland','ukraine':'Ukraine','hungary':'Hungary',
    'slovakia':'Slovakia','slovenia':'Slovenia','iceland':'Iceland',
    'bosnia-herzegovina':'Bosnia & Herzegovina',
    'bosnia & herzegovina':'Bosnia & Herzegovina',
    'argentina':'Argentina','brazil':'Brazil','uruguay':'Uruguay',
    'colombia':'Colombia','chile':'Chile','ecuador':'Ecuador',
    'paraguay':'Paraguay','peru':'Peru','bolivia':'Bolivia','venezuela':'Venezuela',
    'usa':'United States','united states':'United States',
    'mexico':'Mexico','canada':'Canada','costa rica':'Costa Rica',
    'honduras':'Honduras','jamaica':'Jamaica','panama':'Panama',
    'trinidad and tobago':'Trinidad & Tobago','trinidad & tobago':'Trinidad & Tobago',
    'haiti':'Haiti','cuba':'Cuba','el salvador':'El Salvador',
    'curacao':'Curacao','curaçao':'Curacao',
    'morocco':'Morocco','senegal':'Senegal','egypt':'Egypt',
    'nigeria':'Nigeria','cameroon':'Cameroon','ghana':'Ghana',
    'algeria':'Algeria','tunisia':'Tunisia','south africa':'South Africa',
    'ivory coast':"Cote d'Ivoire","cote d'ivoire":"Cote d'Ivoire",
    'côte d\'ivoire':"Cote d'Ivoire",'côte d’ivoire':"Cote d'Ivoire",
    'congo dr':'Congo DR','dr congo':'Congo DR','zaire':'Congo DR',
    'angola':'Angola','togo':'Togo','cape verde':'Cabo Verde',
    'cabo verde':'Cabo Verde',
    'japan':'Japan','south korea':'South Korea','korea republic':'South Korea',
    'iran':'IR Iran','ir iran':'IR Iran','saudi arabia':'Saudi Arabia',
    'australia':'Australia','qatar':'Qatar','china':'China',
    'united arab emirates':'United Arab Emirates','uae':'United Arab Emirates',
    'kuwait':'Kuwait','iraq':'Iraq','north korea':'North Korea',
    'jordan':'Jordan','uzbekistan':'Uzbekistan',
    'new zealand':'New Zealand','tahiti':'Tahiti',
}

def norm(name):
    return TEAM_MAP.get(name.strip().lower(), name.strip())


def detect_date(line, year):
    """
    从行首检测日期，返回 (date_iso, rest_of_line) 或 (None, line).

    支持格式:
      "Fri Jun 9  ..."     → 'YYYY-06-09'
      "9 June  ..."        → 'YYYY-06-09'
      "June 18"            → 'YYYY-06-18'  (1994, 日期占整行)
      "Fri Jun 9"          → 'YYYY-06-09'  (日期占整行)
    """
    # 模式 1: Weekday Month day
    m = re.match(r'([A-Z][a-z]{2})\s+([A-Z][a-z]{2,})\s+(\d{1,2})\b', line)
    if m:
        mon = MONTH_MAP.get(m.group(2).lower()[:3], '01')
        day = m.group(3).zfill(2)
        rest = line[m.end():]
        return f"{year}-{mon}-{day}", rest

    # 模式 2: day Month (数字开头)
    m = re.match(r'(\d{1,2})\s+([A-Z][a-z]+)\b', line)
    if m:
        mon = MONTH_MAP.get(m.group(2).lower(), '01')
        day = m.group(1).zfill(2)
        rest = line[m.end():]
        return f"{year}-{mon}-{day}", rest

    # 模式 3: Month day (1994 格式, "June 18" 占整行)
    m = re.match(r'([A-Z][a-z]+)\s+(\d{1,2})\s*$', line.strip())
    if m:
        mon = MONTH_MAP.get(m.group(1).lower(), '01')
        day = m.group(2).zfill(2)
        return f"{year}-{mon}-{day}", ''

    return None, line


def extract_match(line):
    """
    从行中提取比赛信息。返回 (home, away, hg, ag) 或 None。

    处理策略: 找到第一个 \d+-\d+ 作为全场比分, 然后从 @ 之前的文本提取队名。
    """
    # 去除时间前缀和时区
    s = re.sub(r'^\s*\d{1,2}:\d{2}\s*(?:UTC[+-]\d+\s*)?', '', line).strip()

    # 找第一个 X-Y
    score_m = re.search(r'(\d+)\s*-\s*(\d+)', s)
    if not score_m:
        return None

    hg = int(score_m.group(1))
    ag = int(score_m.group(2))
    before = s[:score_m.start()].strip()
    after = s[score_m.end():].strip()

    # 在 before 中确定主队名
    # 如果有 "v" 分隔符, home = before "v" 的部分
    # 否则 home = before 的全部
    if ' v ' in before:
        home = norm(before.rsplit(' v ', 1)[0].strip())
    else:
        home = norm(before.strip())

    # 在 from score to @ 之间找客队名
    # 去掉 "@ Venue" 之后的内容
    if '@' in after:
        after = after.split('@')[0]

    # 清理: 去掉 a.e.t., (ht), pen. 等
    after = re.sub(r'\ba\.e\.t\.?\b', '', after, flags=re.IGNORECASE)
    after = re.sub(r'\bet\b', '', after, flags=re.IGNORECASE)
    # 去掉 (半场比分) 含嵌套的
    after = re.sub(r'\([\d\-\,\s]+\)', '', after)
    # 去掉 ", X-Y pen." 点球比分
    after = re.sub(r',\s*\d+\s*-\s*\d+\s*pen\.?', '', after, flags=re.IGNORECASE)
    after = re.sub(r'\d+\s*-\s*\d+\s+pen\.?', '', after, flags=re.IGNORECASE)
    after = re.sub(r',?\s*\d+\s*-\s*\d+\s+on\s+penalties', '', after, flags=re.IGNORECASE)
    # 去掉注释
    after = re.sub(r'#.*$', '', after)
    # trim
    after = after.strip().strip(',').strip()

    # 找最后一个有效队名（去掉括号内容后最长的连续文本段）
    # "Brazil v Croatia" → away 在 after 中
    # "Germany" 形式
    # 取空格分隔的连续单词作为候选

    if ' v ' in after:
        away = norm(after.split(' v ', 1)[1].strip() if ' v ' in after.split(' v ', 1) else after.strip())
    else:
        # after 长得像: "Chile" 或 "Costa Rica"
        # 去掉前后额外空格
        away = norm(after.strip())

    # 验证
    if not home or not away or len(home) < 2 or len(away) < 2:
        return None
    if hg > 25 or ag > 25:
        return None
    if home.lower() == away.lower():
        return None

    return home, away, hg, ag


def parse_file(filepath, year, tournament):
    matches = []
    current_date = None

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for line in lines:
        orig = line.rstrip()
        if not orig.strip():
            continue

        stripped = orig.strip()
        if stripped.startswith('=') or stripped.startswith('#'):
            continue
        if re.match(r'^\s*▪', orig):
            continue
        if re.match(r'^\s*Group [A-L]', orig):
            continue
        if re.match(r'^\s*Matchday', orig):
            continue

        # 日期检测
        date_str, rest = detect_date(orig, year)
        if date_str:
            current_date = date_str
            if rest and re.search(r'\d+-\d+', rest) and '@' in rest:
                m = extract_match(rest)
                if m:
                    matches.append([current_date, m[0], m[1], m[2], m[3], tournament, 'Y'])
            continue

        # 比赛行
        if re.search(r'\d+-\d+', orig) and '@' in orig and current_date:
            m = extract_match(orig)
            if m:
                matches.append([current_date, m[0], m[1], m[2], m[3], tournament, 'Y'])

    return matches


def main():
    BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'date'))

    world_cups = [
        ('1990--italy','1990'),('1994--usa','1994'),('1998--france','1998'),
        ('2002--south-korea-n-japan','2002'),('2006--germany','2006'),
        ('2010--south-africa','2010'),('2014--brazil','2014'),('2018--russia','2018'),
        ('2022--qatar','2022'),
    ]

    all_matches = []
    for dirname, year in world_cups:
        n = 0
        for fname in ['cup.txt', 'cup_finals.txt']:
            fp = os.path.join(BASE, dirname, fname)
            if os.path.exists(fp):
                m = parse_file(fp, year, 'FIFA World Cup')
                n += len(m)
                all_matches.extend(m)
        print(f'  {dirname}: {n} 场')

    # 去重
    all_matches.sort(key=lambda x: (x[0], x[1], x[2]))
    unique, seen = [], set()
    for m in all_matches:
        key = (m[0], m[1], m[2], m[3], m[4])
        if key not in seen:
            seen.add(key); unique.append(m)

    print(f'\n总计: {len(unique)} 场 (去重后)')

    # 期望值: 1990(52) + 1994(52) + 1998(64) + 2002(64) + 2006(64) + 2010(64) + 2014(64) + 2018(64) + 2022(64) = 552

    OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'match_history.csv')
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date','home_team','away_team','home_goals','away_goals','tournament','neutral'])
        w.writerows(unique)

    tc = Counter()
    for m in unique: tc[m[1]] += 1; tc[m[2]] += 1
    print(f'涉及球队: {len(tc)}')
    low = [(t,c) for t,c in tc.most_common() if c < 20]
    if low:
        print(f'不足20场: {len(low)} (仅World Cup数据，需要补充更多赛事)')

    print(f'\n[OK] → {OUT}')


if __name__ == '__main__':
    main()
