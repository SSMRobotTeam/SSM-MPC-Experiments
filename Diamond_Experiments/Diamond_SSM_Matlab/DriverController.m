function DriverController(socket)
% tic
    global posi
    % 获取当前时间（秒）
    current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
    timestamp = char(current_time);  % 将 datetime 转换为字符串
    % 构造消息数据
    message = struct();
    message.command = 'set_motors';
    message.timestamp = timestamp;

    message.motorsCount = 3;
    message.isCtlModeUnified = 1;

    message.param.ctlMode = [0,0,0];
    message.param.posiSp  = posi;
    message.param.velcSp  = [0,0,0];
    message.param.forceSp = [0,0,0];
    
    % 编码为 JSON 字符串
    jsonStr = jsonencode(message);
    
    % 发送消息（带话题）
    topic = '/controller/setActuatorDate';
    socket.send(topic, org.zeromq.ZMQ.SNDMORE);  % 发送话题
    socket.send(jsonStr);  % 发送消息
    
    disp(['Sent @ ' message.timestamp]);
    disp(['position= ', num2str(posi)]);
% toc
end
