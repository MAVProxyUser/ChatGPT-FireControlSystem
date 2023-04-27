import dynamixel_sdk as dynamixel

PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
DEVICE_NAME = "/dev/ttyUSB0"  # Change to the correct port name for your setup

DXL_ID_PAN = 1
DXL_ID_TILT = 2
TORQUE_ENABLE_ADDR = 64

# Initialize PortHandler instance
portHandler = dynamixel.PortHandler(DEVICE_NAME)

# Initialize PacketHandler instance
packetHandler = dynamixel.PacketHandler(PROTOCOL_VERSION)

# Open the port
if not portHandler.openPort():
    print("Failed to open the port.")
    exit()

# Set the port baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate.")
    exit()

def read_servo_position(dxl_id):
    ADDR_PRESENT_POSITION = 132
    LEN_PRESENT_POSITION = 4
    return packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)

def move_servo(dxl_id, goal_position):
    ADDR_GOAL_POSITION = 116
    LEN_GOAL_POSITION = 4
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, goal_position)

def set_servo_torque(servo_id, enable):
    ADDR_TORQUE_ENABLE = 64
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    if enable:
        packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    else:
        packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)


