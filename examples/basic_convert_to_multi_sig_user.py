import example_utils

from hyperliquid.utils import constants


def main():
    address, info, exchange = example_utils.setup(constants.TESTNET_API_URL, skip_ws=True)

    if exchange.account_address != exchange.wallet.address:
        raise Exception("Agents do not have permission to convert to multi-sig user")

    # Authorized users are the public addresses of the wallets that will be able to sign on behalf of the multi-sig user.
    # Some additional notes:
    # Only existing users may be used. In other words, the authorized users must have deposited. Otherwise this conversion will fail.
    # The multi-sig signatures must be generated by the authorized user's wallet. Agent/API wallets cannot be used.
    authorized_user_1 = "0x0000000000000000000000000000000000000000"
    authorized_user_2 = "0x0000000000000000000000000000000000000001"
    threshold = 1
    convert_result = exchange.convert_to_multi_sig_user([authorized_user_1, authorized_user_2], threshold)
    print(convert_result)


if __name__ == "__main__":
    main()
