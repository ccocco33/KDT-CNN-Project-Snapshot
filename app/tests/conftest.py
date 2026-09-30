def pytest_collection_modifyitems(items):
    """docstring 첫 줄을 테스트 표시 이름으로 사용
    - docstring 없는 테스트는 함수 이름 그대로
    - parametrize 인자는 [id] 로 붙임
    """
    for item in items:
        doc = (item.function.__doc__ or "").strip()
        if doc:
            name = doc.splitlines()[0]
            param = f"[{item.callspec.id}]" if hasattr(item, "callspec") else ""
            item._nodeid = f"{item.nodeid.split('::')[0]}::{name}{param}"
