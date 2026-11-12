import flet.testing as ftt


async def test_app_loads(flet_app: ftt.FletTestApp):
    """冒烟测试：页面能加载、不抛异常即可。"""
    tester = flet_app.tester
    await tester.pump_and_settle()
