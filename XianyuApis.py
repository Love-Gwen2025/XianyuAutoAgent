import time
import os
import re

import requests
from loguru import logger
from utils.xianyu_utils import generate_sign


class CookieExpiredError(Exception):
    """Cookie 过期或失效异常，需要用户手动更新"""
    pass


class XianyuApis:
    def __init__(self):
        self.url = os.getenv("GOOFISH_API_URL", "https://h5api.m.goofish.com/h5/mtop.taobao.idlemessage.pc.login.token/1.0/")
        self.app_key = os.getenv("GOOFISH_APP_KEY", "34839810")
        self.ws_app_key = os.getenv("GOOFISH_WS_APP_KEY", "444e9908a51d1cb236a27862abc769c9")
        self.headless = os.getenv("HEADLESS", "True").lower() == "true"
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'no-cache',
            'origin': 'https://www.goofish.com',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://www.goofish.com/',
            'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        })

    def clear_duplicate_cookies(self):
        """清理重复的cookies"""
        # 创建一个新的CookieJar
        new_jar = requests.cookies.RequestsCookieJar()

        # 记录已经添加过的cookie名称
        added_cookies = set()

        # 按照cookies列表的逆序遍历（最新的通常在后面）
        cookie_list = list(self.session.cookies)
        cookie_list.reverse()

        for cookie in cookie_list:
            # 如果这个cookie名称还没有添加过，就添加到新jar中
            if cookie.name not in added_cookies:
                new_jar.set_cookie(cookie)
                added_cookies.add(cookie.name)

        # 替换session的cookies
        self.session.cookies = new_jar

        # 更新完cookies后，更新.env文件
        self.update_env_cookies()

    def update_env_cookies(self):
        """更新.env文件中的COOKIES_STR"""
        try:
            # 获取当前cookies的字符串形式
            cookie_str = '; '.join([f"{cookie.name}={cookie.value}" for cookie in self.session.cookies])

            # 读取.env文件
            env_path = os.path.join(os.getcwd(), '.env')
            if not os.path.exists(env_path):
                logger.warning(".env文件不存在，无法更新COOKIES_STR")
                return

            with open(env_path, 'r', encoding='utf-8') as f:
                env_content = f.read()

            # 使用正则表达式替换COOKIES_STR的值
            if 'COOKIES_STR=' in env_content:
                new_env_content = re.sub(
                    r'COOKIES_STR=.*',
                    f'COOKIES_STR={cookie_str}',
                    env_content
                )

                # 写回.env文件
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(new_env_content)

                logger.debug("已更新.env文件中的COOKIES_STR")
            else:
                logger.warning(".env文件中未找到COOKIES_STR配置项")
        except Exception as e:
            logger.warning(f"更新.env文件失败: {str(e)}")

    def hasLogin(self, retry_count=0):
        """调用hasLogin.do接口进行登录状态检查"""
        if retry_count >= 2:
            logger.error("Login检查失败，重试次数过多")
            return False

        try:
            url = 'https://passport.goofish.com/newlogin/hasLogin.do'
            params = {
                'appName': 'xianyu',
                'fromSite': '77'
            }
            data = {
                'hid': self.session.cookies.get('unb', ''),
                'ltl': 'true',
                'appName': 'xianyu',
                'appEntrance': 'web',
                '_csrf_token': self.session.cookies.get('XSRF-TOKEN', ''),
                'umidToken': '',
                'hsiz': self.session.cookies.get('cookie2', ''),
                'bizParams': 'taobaoBizLoginFrom=web',
                'mainPage': 'false',
                'isMobile': 'false',
                'lang': 'zh_CN',
                'returnUrl': '',
                'fromSite': '77',
                'isIframe': 'true',
                'documentReferer': 'https://www.goofish.com/',
                'defaultView': 'hasLogin',
                'umidTag': 'SERVER',
                'deviceId': self.session.cookies.get('cna', '')
            }

            response = self.session.post(url, params=params, data=data)
            res_json = response.json()

            if res_json.get('content', {}).get('success'):
                logger.debug("Login成功")
                # 清理和更新cookies
                self.clear_duplicate_cookies()
                return True
            else:
                logger.warning(f"Login失败: {res_json}")
                time.sleep(0.5)
                return self.hasLogin(retry_count + 1)

        except Exception as e:
            logger.error(f"Login请求异常: {str(e)}")
            time.sleep(0.5)
            return self.hasLogin(retry_count + 1)

    def get_token(self, device_id):
        max_retries = 3
        has_relogged = False

        for attempt in range(max_retries):
            params = {
                'jsv': '2.7.2',
                'appKey': self.app_key,
                't': str(int(time.time()) * 1000),
                'sign': '',
                'v': '1.0',
                'type': 'originaljson',
                'accountSite': 'xianyu',
                'dataType': 'json',
                'timeout': '20000',
                'api': 'mtop.taobao.idlemessage.pc.login.token',
                'sessionOption': 'AutoLoginOnly',
                'spm_cnt': 'a21ybx.im.0.0',
            }
            data_val = '{"appKey":"' + self.ws_app_key + '","deviceId":"' + device_id + '"}'
            data = {
                'data': data_val,
            }

            token = self.session.cookies.get('_m_h5_tk', '').split('_')[0]
            sign = generate_sign(params['t'], token, data_val)
            params['sign'] = sign

            try:
                response = self.session.post(self.url, params=params, data=data)
                res_json = response.json()

                if isinstance(res_json, dict):
                    ret_value = res_json.get('ret', [])
                    if any('SUCCESS::调用成功' in ret for ret in ret_value):
                        logger.info("Token获取成功")
                        return res_json

                    # 检测风控/限流错误
                    error_msg = str(ret_value)
                    if 'RGV587_ERROR' in error_msg or '被挤爆啦' in error_msg:
                        logger.error(f"触发风控: {ret_value}")
                        logger.error("系统目前无法自动解决，请进入闲鱼网页版-点击消息-过滑块-复制最新的Cookie")

                        if self.headless:
                            raise CookieExpiredError("触发风控，headless 模式下无法交互输入，请更新 COOKIES_STR 后重启")

                        print("\n" + "="*50)
                        new_cookie_str = input("请输入新的Cookie字符串 (复制浏览器中的完整cookie，直接回车则退出程序): ").strip()
                        print("="*50 + "\n")

                        if new_cookie_str:
                            try:
                                from http.cookies import SimpleCookie
                                cookie = SimpleCookie()
                                cookie.load(new_cookie_str)

                                self.session.cookies.clear()
                                for key, morsel in cookie.items():
                                    self.session.cookies.set(key, morsel.value, domain='.goofish.com')

                                logger.success("Cookie已更新，正在尝试重连...")
                                self.update_env_cookies()

                                # 重置重试计数从头开始
                                attempt = -1  # 下次循环变为0
                                continue
                            except CookieExpiredError:
                                raise
                            except Exception as e:
                                logger.error(f"Cookie解析失败: {e}")
                                raise CookieExpiredError(f"Cookie解析失败: {e}")
                        else:
                            logger.info("用户取消输入，程序退出")
                            raise CookieExpiredError("用户取消输入")

                    logger.warning(f"Token API调用失败，错误信息: {ret_value}")
                    if 'Set-Cookie' in response.headers:
                        logger.debug("检测到Set-Cookie，更新cookie")
                        self.clear_duplicate_cookies()
                else:
                    logger.error(f"Token API返回格式异常: {res_json}")

            except CookieExpiredError:
                raise
            except Exception as e:
                logger.error(f"Token API请求异常: {str(e)}")

            backoff = min(0.5 * (2 ** attempt), 30)
            logger.debug(f"Token重试等待 {backoff:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(backoff)

        # 所有重试失败，尝试重新登录（仅一次）
        if not has_relogged:
            logger.warning("获取token失败，尝试重新登录")
            has_relogged = True
            if self.hasLogin():
                logger.info("重新登录成功，重新尝试获取token")
                return self.get_token(device_id)

        logger.error("重新登录失败，Cookie已失效")
        logger.error("请更新.env文件中的COOKIES_STR后重新启动")
        raise CookieExpiredError("Cookie已失效，请更新COOKIES_STR后重新启动")

    def get_item_info(self, item_id):
        """获取商品信息，自动处理token失效的情况"""
        max_retries = 3

        for attempt in range(max_retries):
            params = {
                'jsv': '2.7.2',
                'appKey': self.app_key,
                't': str(int(time.time()) * 1000),
                'sign': '',
                'v': '1.0',
                'type': 'originaljson',
                'accountSite': 'xianyu',
                'dataType': 'json',
                'timeout': '20000',
                'api': 'mtop.taobao.idle.pc.detail',
                'sessionOption': 'AutoLoginOnly',
                'spm_cnt': 'a21ybx.im.0.0',
            }

            data_val = '{"itemId":"' + item_id + '"}'
            data = {
                'data': data_val,
            }

            token = self.session.cookies.get('_m_h5_tk', '').split('_')[0]
            sign = generate_sign(params['t'], token, data_val)
            params['sign'] = sign

            try:
                response = self.session.post(
                    'https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/',
                    params=params,
                    data=data
                )

                res_json = response.json()
                if isinstance(res_json, dict):
                    ret_value = res_json.get('ret', [])
                    if any('SUCCESS::调用成功' in ret for ret in ret_value):
                        logger.debug(f"商品信息获取成功: {item_id}")
                        return res_json

                    logger.warning(f"商品信息API调用失败，错误信息: {ret_value}")
                    if 'Set-Cookie' in response.headers:
                        logger.debug("检测到Set-Cookie，更新cookie")
                        self.clear_duplicate_cookies()
                else:
                    logger.error(f"商品信息API返回格式异常: {res_json}")

            except Exception as e:
                logger.error(f"商品信息API请求异常: {str(e)}")

            backoff = min(0.5 * (2 ** attempt), 30)
            logger.debug(f"商品信息重试等待 {backoff:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(backoff)

        logger.error("获取商品信息失败，重试次数过多")
        return {"error": "获取商品信息失败，重试次数过多"}
