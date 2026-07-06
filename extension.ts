import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import FormData = require('form-data');

export function activate(context: vscode.ExtensionContext) {
    console.log('✅ Extension "Voice IDE Assistant" đã khởi động!');

    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'voice-controller-ide.listen';
    statusBarItem.text = '$(mic) Bật Mic (Ra lệnh)';
    statusBarItem.tooltip = 'Bấm vào đây để nói lệnh điều khiển VS Code';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    let disposable = vscode.commands.registerCommand('voice-controller-ide.listen', async () => {
        vscode.window.showInformationMessage('🎙️ Đang lắng nghe (3 giây)... Hãy nói!');

        const extPath = context.extensionPath;
        const recordScript = path.join(extPath, 'record.py');
        const audioFile = path.join(extPath, 'command.wav');

        try {
            cp.execSync(`python "${recordScript}" "${audioFile}"`);
        } catch (err) {
            vscode.window.showErrorMessage('❌ Lỗi ghi âm. Đảm bảo máy đã cài thư viện sounddevice.');
            return;
        }

        vscode.window.showInformationMessage('⏳ Đang nhờ AI phân tích...');
        try {
            const formData = new FormData();
            formData.append('file', fs.createReadStream(audioFile));

            const response = await axios.post('http://127.0.0.1:8000/api/voice-command', formData, {
                headers: formData.getHeaders()
            });

            const data = response.data;
            if (data.status === 'success') {
                vscode.window.showInformationMessage(`🗣️ Bạn nói: "${data.recognized_text}"`);

                if (data.command_id !== "UNKNOWN_COMMAND") {
                    vscode.commands.executeCommand(data.command_id);
                    vscode.window.showInformationMessage(`✅ Đã thực thi lệnh: ${data.command_id}`);
                } else {
                    vscode.window.showWarningMessage('❓ AI hiểu bạn nói gì, nhưng không tìm thấy lệnh VS Code tương ứng.');
                }
            } else {
                vscode.window.showErrorMessage(`❌ Lỗi từ Backend: ${data.message}`);
            }
        } catch (error) {
            vscode.window.showErrorMessage('❌ Không thể kết nối tới Backend AI. Bạn đã chạy file main.py chưa?');
        }
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}