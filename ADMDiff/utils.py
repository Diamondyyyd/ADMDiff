import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

import json
from torch import nn


def train(
    model,
    config,
    train_loader,
    test_loader1=None,
    test_loader2=None,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    nsample_list = [
        1
    ]
    best_mse_score = 10000
    stop_counter = 0
    best_valid_loss = 1e10
    # for epoch_no in range(config["epochs"]):
    # !for test!
    for epoch_no in range(0,500):

        avg_loss = 0
        model.train()
        with tqdm(train_loader) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                # if batch_no == 100:
                #   break
                optimizer.zero_grad()
                #!for test
                # if batch_no == 100:
                #     break

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

    
        mse_score = validation(model, valid_loader ,nsample=1)
        if mse_score < best_mse_score:
            stop_counter = 0
            best_mse_score = mse_score
            print("best mse score update")
            print("now best is")
            print(best_mse_score)
            output_path = foldername + f"/best-model.pth"
            torch.save(model.state_dict(), output_path)
        else:
            stop_counter += 1

        if stop_counter > 10:
            break

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def validation(model, valid_loader, nsample=20, scaler=1):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)  # 保持维度一致性
                
                # 获取趋势分量并确保在正确的设备上
                trend = test_batch["trend"].unsqueeze(1)  # 获取趋势分量
                trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上
                
                # 将趋势分量加回重建结果
                samples = samples + trend
                
                samples_median = samples.median(dim=1)
                
                # 计算MSE和MAE时使用eval_points
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()  # 使用eval_points的和作为评估点数量

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
    return np.sqrt(mse_total / evalpoints_total)

# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
def evaluate(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # 计算strategy1的结果
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)  # 保持维度一致性
                trend = test_batch["trend"].unsqueeze(1)  # 获取趋势分量
                trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上

                output2 = model.evaluate(next(test_loader2), nsample)
                samples2 = output2[0]
                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                
                # 合并两个策略的稳定分量
                samples_length = samples.shape[2]
                samples[:,:,samples_length // 4 : samples_length //2, :] = samples2[:,:,samples_length // 4 : samples_length //2, :]
                samples[:,:,samples_length - samples_length // 4:,:] = samples2[:,:,samples_length - samples_length // 4:,:]

                # 将趋势分量加回重建结果
                samples = samples + trend
                
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # 计算MSE和MAE时使用eval_points
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()  # 使用eval_points的和作为评估点数量

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + f"/{epoch_number}-result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)


def window_trick_evaluate(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name="",split=4):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):


                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                trend = test_batch["trend"]  # 获取趋势分量
                trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上

                output2 = model.evaluate(next(test_loader2), nsample)
                samples2 = output2[0]
                eval_points2 = output2[2]

                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                eval_points2 = eval_points2.permute(0,2,1)

                samples = samples.squeeze() * eval_points + samples2.squeeze() * eval_points2

                samples_length = samples.shape[1]

                if batch_no == 1:
                    head = samples[0, 0 : samples_length // split, :]
                    print("shape of head is")
                    print(head.shape)
                    head_c_target = c_target[0, 0 : samples_length // split , :]
                    print("shape of head c target is")
                    print(head_c_target.shape)
                    head_observed_points = observed_points[0, 0 : samples_length // split , :]
                    print("shape of observed points is")
                    print(head_observed_points.shape)
                    # 保存头部趋势分量
                    head_trend = trend[0, 0 : samples_length // split, :]
                    
                    # 确保所有头部数据在同一设备上
                    device = head.device
                    head = head.to(device)
                    head_c_target = head_c_target.to(device)
                    head_observed_points = head_observed_points.to(device)
                    head_trend = head_trend.to(device)

                samples = samples[:, samples_length // split: samples_length - samples_length // split, :]
                c_target = c_target[:, samples_length // split: samples_length - samples_length // split, :]
                observed_points = observed_points[:, samples_length // split: samples_length - samples_length // split, :]
                trend = trend[:, samples_length // split: samples_length - samples_length // split, :]

                eval_points = torch.ones_like(samples)
                
                # 将趋势分量加回重建结果
                samples_with_trend = samples + trend

                print("shape of samples is")
                print(samples.shape)
                print("shape of c target is")
                print(c_target.shape)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples_with_trend)

                mse_current = (
                    ((samples_with_trend - c_target)) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_with_trend - c_target))
                ) * scaler


                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += torch.ones_like(samples).sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

                # if np.sqrt(mse_total / evalpoints_total) > 20:
                #     residual = ((samples_median.values - c_target)) ** 2 * (scaler ** 2)
                #     print(residual)

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                head = head.to("cpu")
                head_c_target = head_c_target.to("cpu")
                head_observed_points = head_observed_points.to("cpu")
                head_trend = head_trend.to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        head + head_trend,
                        head_c_target,
                        head_observed_points,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )




def middle_evaluate(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:

            middle_total_list = [0] * 100

            for batch_no, test_batch in enumerate(it, start=1):

                output = model.get_middle_evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time,middle_results_samples = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                middle_results_samples = middle_results_samples.permute(0,1,3,2)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                trend = test_batch["trend"].unsqueeze(1)  # 获取趋势分量
                trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上

                output2 = model.get_middle_evaluate(next(test_loader2), nsample)
                samples2 = output2[0]
                middle_results_samples2 = output2[-1]


                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                middle_results_samples2 = middle_results_samples2.permute(0,1,3,2)

                samples_length = samples.shape[2]
                samples[:,:,samples_length // 4 : samples_length //2, :] = samples2[:,:,samples_length // 4 : samples_length //2, :]
                samples[:,:,samples_length - samples_length // 4:,:] = samples2[:,:,samples_length - samples_length // 4:,:]

                middle_results_samples[:,:,samples_length // 4 : samples_length //2, :] = middle_results_samples2[:,:,samples_length // 4 : samples_length //2, :]
                middle_results_samples[:,:,samples_length - samples_length // 4:,:] = middle_results_samples2[:,:,samples_length - samples_length // 4:,:]

                # 将趋势分量加回重建结果
                samples = samples + trend
                
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target)) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target))
                ) * scaler

                for i in range(0,100):
                    middle_results_with_trend = middle_results_samples[:,i,:,:].to("cpu") + trend.to("cpu")
                    middle_mse = (
                    ((middle_results_with_trend - c_target.to("cpu"))) ** 2
                ) * (scaler ** 2)
                    middle_total_list[i] += middle_mse.sum()
                    print("middle mse is:")
                    print(middle_mse.sum())

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                # evalpoints_total += eval_points.sum().item()
                evalpoints_total += torch.ones_like(mse_current).sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )


            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + f"/{epoch_number}-result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

            print("final construction error")
            for item in middle_total_list:
                print(item)

def window_trick_evaluate_middle(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name="",stop_number=-1,split=4):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_middle = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                if batch_no == stop_number:
                    break

                output = model.get_middle_evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time, middle_result = output
                trend = test_batch["trend"]  # 获取趋势分量
                trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上

                print("shape of middle result is")
                print(middle_result.shape)
                middle_result = middle_result.permute(0, 1, 3, 2)
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # 计算strategy2的结果
                test_batch2 = next(test_loader2)
                output2 = model.get_middle_evaluate(test_batch2, nsample)
                samples2 = output2[0]
                eval_points2 = output2[2].permute(0,2,1)
                trend2 = test_batch2["trend"]  # 获取strategy2的趋势分量
                trend2 = trend2.to(samples2.device)  # 确保trend2在与samples2相同的设备上

                middle_result2 = output2[-1]
                middle_result2 = middle_result2.permute(0, 1, 3, 2)

                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                # samples_length = samples.shape[2]

                # 合并两个策略的稳定分量和趋势分量
                samples = samples.squeeze() * eval_points + samples2.squeeze() * eval_points2
                trend = trend * eval_points + trend2 * eval_points2

                samples_length = samples.shape[1]
                print(f"shape of samples after merging is {samples.shape}")
                print(f"shape of middle result before merging is {middle_result.shape}")
                print(f"shape of eval points is {eval_points.shape}")

                middle_result = middle_result.cpu() * eval_points.unsqueeze(1).cpu() + middle_result2.cpu() * eval_points2.unsqueeze(1).cpu()

                print("shape of middle result after merging")
                print(middle_result.shape)

                # 保存头条带
                if batch_no == 1:
                    head = samples[0, 0 : samples_length // split, :]
                    print("shape of head is")
                    print(head.shape)
                    head_c_target = c_target[0, 0 : samples_length // split , :]
                    print("shape of head c target is")
                    print(head_c_target.shape)
                    head_observed_points = observed_points[0, 0 : samples_length // split , :]
                    print("shape of observed points is")
                    print(head_observed_points.shape)
                    head_middle = middle_result[0, :, 0 : samples_length // split, :]
                    head_trend = trend[0, 0 : samples_length // split, :]  # 保存头部趋势分量
                    
                    # 确保所有头部数据在同一设备上
                    device = head.device
                    head = head.to(device)
                    head_c_target = head_c_target.to(device)
                    head_observed_points = head_observed_points.to(device)
                    head_middle = head_middle.to(device)
                    head_trend = head_trend.to(device)

                # 截取中间部分
                samples = samples[:, samples_length // split: samples_length - samples_length // split, :]
                c_target = c_target[:, samples_length // split: samples_length - samples_length // split, :]
                observed_points = observed_points[:, samples_length // split: samples_length - samples_length // split, :]
                trend = trend[:, samples_length // split: samples_length - samples_length // split, :]
                middle_result = middle_result[:, :, samples_length // split: samples_length - samples_length // split, :]

                eval_points = torch.ones_like(samples)
                samples_median = samples

                # 只对重建结果加回趋势
                samples_with_trend = samples_median + trend  # 重建结果加回趋势

                # 保存结果
                all_target.append(c_target)  # 目标值就是原始数据，不需要加趋势
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples_with_trend)  # 保存加回趋势后的重建结果
                all_middle.append(middle_result)

                # 计算误差（重建结果已加回趋势，可以直接与原始数据比较）
                mse_current = ((samples_with_trend - c_target)) ** 2 * (scaler ** 2)
                mae_current = torch.abs((samples_with_trend - c_target)) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += torch.ones_like(samples).sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

                # if np.sqrt(mse_total / evalpoints_total) > 20:
                #     residual = ((samples_median.values - c_target)) ** 2 * (scaler ** 2)
                #     print(residual)

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}_stop_number_{stop_number}.pk", "wb"
            ) as f:
                # 确保所有数据在CPU上，以便保存
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                all_middle = torch.cat(all_middle,dim=0).to("cpu")
                
                # 确保head相关数据也在CPU上
                head = head.to("cpu")
                head_c_target = head_c_target.to("cpu")
                head_observed_points = head_observed_points.to("cpu")
                head_trend = head_trend.to("cpu")
                head_middle = head_middle.to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,  # 已经加回趋势的重建结果
                        all_target,             # 原始数据（目标值）
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        head + head_trend,      # 头部重建结果加回趋势
                        head_c_target,          # 头部原始数据（目标值）
                        head_observed_points,
                        scaler,
                        mean_scaler,
                        all_middle,
                        head_middle
                    ],
                    f,
                )





def ddim_evaluate(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name="",ddim_eta=0,ddim_steps=10):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                # output = model.evaluate(test_batch, nsample)
                output = model.ddim_evaluate(test_batch, nsample, ddim_eta=ddim_eta, ddim_steps=ddim_steps)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                trend = test_batch["trend"].unsqueeze(1)  # 获取趋势分量
                trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上

                # output2 = model.evaluate(next(test_loader2), nsample)
                output2 = model.ddim_evaluate(next(test_loader2), nsample, ddim_eta=ddim_eta, ddim_steps=ddim_steps)
                samples2 = output2[0]

                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples_length = samples.shape[2]
                samples[:,:,samples_length // 4 : samples_length //2, :] = samples2[:,:,samples_length // 4 : samples_length //2, :]
                samples[:,:,samples_length - samples_length // 4:,:] = samples2[:,:,samples_length - samples_length // 4:,:]

                # 将趋势分量加回重建结果
                samples = samples + trend
                
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target)) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target))
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                # evalpoints_total += eval_points.sum().item()
                evalpoints_total += torch.ones_like(mse_current).sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + f"/{epoch_number}-result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)


def ensemble(model, test_loader, nsample=10, scaler=1, name = ""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        impute_sample_dict = {}

        for i in tqdm(range(0,nsample)):
            impute_sample_dict[i] = {}
            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = model.evaluate(test_batch, 1)
                    samples, c_target, eval_points, observed_points, observed_time = output
                    samples = samples.permute(0, 1, 3, 2).squeeze()  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)  # 保持维度一致性
                    trend = test_batch["trend"]  # 获取趋势分量
                    trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上
                    
                    # 将趋势分量加回重建结果
                    samples = samples + trend
                    
                    samples_median = samples
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    # 计算MSE和MAE时使用eval_points
                    mse_current = (
                        ((samples_median - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    mae_current = (
                        torch.abs((samples_median - c_target) * eval_points)
                    ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()  # 使用eval_points的和作为评估点数量

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                if 1:
                    all_target = torch.cat(all_target, dim=0).to("cpu")
                    all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                    all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                    all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                    all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                    impute_sample_dict[i]["all_target"] = all_target
                    impute_sample_dict[i]["all_evalpoint"] = all_evalpoint
                    impute_sample_dict[i]["all_observed_point"] = all_observed_point
                    impute_sample_dict[i]["all_observed_time"] = all_observed_time
                    impute_sample_dict[i]["all_generated_samples"] = all_generated_samples
    torch.save(impute_sample_dict,name)

def ddim_ensemble(model, test_loader, nsample=10, scaler=1, name = "",ddim_eta=1,ddim_steps =10):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        impute_sample_dict = {}

        for i in tqdm(range(0,nsample)):
            impute_sample_dict[i] = {}
            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = model.ddim_evaluate(test_batch, 1,ddim_eta=ddim_eta,ddim_steps = ddim_steps)
                    samples, c_target, eval_points, observed_points, observed_time = output
                    samples = samples.permute(0, 1, 3, 2).squeeze()  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)  # 保持维度一致性
                    trend = test_batch["trend"]  # 获取趋势分量
                    trend = trend.to(samples.device)  # 确保trend在与samples相同的设备上
                    
                    # 将趋势分量加回重建结果
                    samples = samples + trend
                    
                    samples_median = samples
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    # 计算MSE和MAE时使用eval_points
                    mse_current = (
                        ((samples_median - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    mae_current = (
                        torch.abs((samples_median - c_target) * eval_points)
                    ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()  # 使用eval_points的和作为评估点数量

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                if 1:
                    all_target = torch.cat(all_target, dim=0).to("cpu")
                    all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                    all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                    all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                    all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                    impute_sample_dict[i]["all_target"] = all_target
                    impute_sample_dict[i]["all_evalpoint"] = all_evalpoint
                    impute_sample_dict[i]["all_observed_point"] = all_observed_point
                    impute_sample_dict[i]["all_observed_time"] = all_observed_time
                    impute_sample_dict[i]["all_generated_samples"] = all_generated_samples
    torch.save(impute_sample_dict,name)



def evaluate_stable_loss(model, dataloader, foldername="", name=""):
    """
    评估动态分解的稳定分量与预处理稳定分量之间的差异
    
    参数:
        model: 模型
        dataloader: 数据加载器
        foldername: 结果保存文件夹
        name: 实验名称
    
    返回:
        平均稳定分量损失
    """
    with torch.no_grad():
        model.eval()
        
        # 累计损失
        total_stable_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="评估稳定分量损失"):
            # 获取真正的稳定分量和预处理稳定分量
            stable = batch['stable'].to(model.device)  # 模型直接预测的稳定分量
            preprocess_stable = batch['preprocess_stable'].to(model.device)
            
            # 调整维度一致性
            if stable.shape != preprocess_stable.shape:
                preprocess_stable = preprocess_stable.permute(0, 2, 1)
                
            # 计算MSE损失
            mse_loss = nn.MSELoss(reduction='mean')(stable, preprocess_stable)
            
            total_stable_loss += mse_loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_stable_loss = total_stable_loss / num_batches
        
        # 保存结果
        if foldername:
            with open(f"{foldername}/stable_loss_{name}.json", "w") as f:
                json.dump({"stable_loss": avg_stable_loss}, f, indent=4)
        
        return avg_stable_loss