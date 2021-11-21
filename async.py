import asyncio
import requests
import time
import nest_asyncio
# def do_request():
#     rsp = requests.get('https://example.com')
#     print(f"result : {rsp.status_code}")


# def main():
#     for _ in range(10):
#         do_request()

# if __name__ == "__main__":
#     main()


# async def sample():
#             await asyncio.sleep(1)
#             print("123213")

# asyncio.run(sample())

class NestTest():
    # def __init__(self) -> None:
    #     self.loop = asyncio.get_event_loop()
        # self.s = s
        # nest_asyncio.apply(self.loop)
        # asyncio.set_event_loop(self.loop)

    async def coro(self, s):
        await asyncio.sleep(0.3)
        print(f"{s}")
        return s

    async def test_nesting(self, s):
        for _ in range(1):
            await self.coro(s)
        return f"result:{s}"

async def wicker():
    t1 = asyncio.create_task(say("123", 2))
    t2 = asyncio.create_task(say("546", 0))
    await t1
    await t2
    print("456")

async def say(phrase, time):
    print(phrase)
    await asyncio.sleep(time)

asyncio.run(wicker())



