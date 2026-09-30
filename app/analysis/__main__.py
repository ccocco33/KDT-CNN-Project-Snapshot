"""로그 분석 실행
- 실행 (app/ 에서)
  - 요약: .venv/bin/python -m analysis [로그 파일 또는 glob ...]
  - 직접 쿼리: .venv/bin/python -m analysis --sql "select ev, count(*) from events group by ev"
- 로그를 지정하지 않으면 logs/*.jsonl 전체
"""
import argparse
import glob
import sys

from .logs import DEFAULT_GLOB, connect, run_reports


def main():
    parser = argparse.ArgumentParser(prog="python -m analysis", description="photo_app 로그 분석")
    parser.add_argument("files", nargs="*", default=[DEFAULT_GLOB], help="로그 파일 또는 glob (기본: logs/*.jsonl)")
    parser.add_argument("--sql", help="직접 실행할 쿼리. 테이블 이름 events")
    args = parser.parse_args()

    if not any(glob.glob(f) for f in args.files):
        sys.exit(f"로그 파일 없음: {' '.join(args.files)}")
    con = connect(args.files)
    if args.sql:
        con.sql(args.sql).show(max_rows=1000)
        return
    for title, result in run_reports(con):
        print(f"\n## {title}")
        if result is None:
            print("(기록 없음)")
        else:
            result.show(max_rows=1000)


if __name__ == "__main__":
    main()
